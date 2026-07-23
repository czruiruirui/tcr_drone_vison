#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

// TensorRT Logger
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false),
  runtime_(nullptr), engine_(nullptr), context_(nullptr),
  stream_(nullptr), buffers_{nullptr, nullptr},
  host_output_buffer_(nullptr), input_size_(0), output_size_(0),
  input_h_(640), input_w_(640), class_num_(9)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  score_threshold_ = yaml["score_threshold"].as<float>(0.3f);
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  flip_about_center_ = yaml["flip_about_center"].as<bool>(false);  // 新增：镜像翻转配置
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // sigmoid 单调递增，parse 中先用 logit 阈值预过滤，避免对全部输出逐行计算 exp
  logit_threshold_ = std::log(score_threshold_ / (1.0 - score_threshold_));

#ifdef USE_TENSORRT
  // 加载 TensorRT engine
  if (!loadEngine(model_path_)) {
    throw std::runtime_error("Failed to load TensorRT engine: " + model_path_);
  }
#else
  throw std::runtime_error("USE_TENSORRT not defined!");
#endif
}

YOLOV5::~YOLOV5()
{
#ifdef USE_TENSORRT
  // 释放资源 - TensorRT 8.6+ 使用 delete 而不是 destroy()
  if (host_input_buffer_) {
    cudaFreeHost(host_input_buffer_);
    host_input_buffer_ = nullptr;
  }
  if (host_output_buffer_) {
    cudaFreeHost(host_output_buffer_);
    host_output_buffer_ = nullptr;
  }
  if (buffers_[0]) {
    cudaFree(buffers_[0]);
    buffers_[0] = nullptr;
  }
  if (buffers_[1]) {
    cudaFree(buffers_[1]);
    buffers_[1] = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (context_) {
    delete context_;
    context_ = nullptr;
  }
  if (engine_) {
    delete engine_;
    engine_ = nullptr;
  }
  if (runtime_) {
    delete runtime_;
    runtime_ = nullptr;
  }
#endif
}

#ifdef USE_TENSORRT
bool YOLOV5::loadEngine(const std::string& engine_path)
{
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        tools::logger()->error("Failed to open engine: {}", engine_path);
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    // 创建 runtime
    static TensorRTLogger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        tools::logger()->error("Failed to create TensorRT runtime");
        return false;
    }
    
    // 反序列化 engine - TensorRT 8.6+ 只有 2 个参数
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        tools::logger()->error("Failed to deserialize TensorRT engine");
        return false;
    }
    
    // 创建执行上下文
    context_ = engine_->createExecutionContext();
    if (!context_) {
        tools::logger()->error("Failed to create TensorRT execution context");
        return false;
    }
    
    // 获取输入输出尺寸 - 使用新的 API
    const char* input_name = engine_->getIOTensorName(0);
    const char* output_name = engine_->getIOTensorName(1);
    
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name);
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_name);
    
    input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; i++) {
        input_size_ *= input_dims.d[i];
    }
    
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; i++) {
        output_size_ *= output_dims.d[i];
    }
    
    tools::logger()->info("YOLOV5 TensorRT Engine loaded:");
    tools::logger()->info("  Input size: {}", input_size_);
    tools::logger()->info("  Output size: {}", output_size_);
    tools::logger()->info("  Input dims: {}", input_dims.nbDims);
    tools::logger()->info("  Output dims: {}", output_dims.nbDims);
    
    // 分配 GPU 内存
    cudaMalloc(&buffers_[0], input_size_ * sizeof(float));
    cudaMalloc(&buffers_[1], output_size_ * sizeof(float));
    cudaStreamCreate(&stream_);
    
    // 分配锁页(pinned)主机内存：H2D/D2H 走 DMA，比可分页内存快得多
    cudaHostAlloc(&host_input_buffer_, input_size_ * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&host_output_buffer_, output_size_ * sizeof(float), cudaHostAllocDefault);

    // 复用的 letterbox 画布与浮点输入，避免每帧重新分配
    letterbox_canvas_ = cv::Mat(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    input_float_ = cv::Mat(input_h_, input_w_, CV_32FC3);
    
    // 保存张量名称供后续使用
    input_tensor_name_ = input_name;
    output_tensor_name_ = output_name;
    
    return true;
}

void YOLOV5::preProcess(const cv::Mat& bgr_img)
{
    // 复用 640x640 画布，灰色填充
    letterbox_canvas_.setTo(cv::Scalar(114, 114, 114));

    auto x_scale = static_cast<double>(input_h_) / bgr_img.rows;
    auto y_scale = static_cast<double>(input_w_) / bgr_img.cols;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(bgr_img.rows * scale);
    auto w = static_cast<int>(bgr_img.cols * scale);

    // 缩放图像并居中放置
    cv::Mat resized;
    cv::resize(bgr_img, resized, {w, h});
    auto roi = cv::Rect((input_w_ - w) / 2, (input_h_ - h) / 2, w, h);
    resized.copyTo(letterbox_canvas_(roi));

    // 转换为 RGB 并归一化到 [0, 1]（写入复用的浮点缓冲）
    cv::cvtColor(letterbox_canvas_, letterbox_canvas_, cv::COLOR_BGR2RGB);
    letterbox_canvas_.convertTo(input_float_, CV_32FC3, 1.0 / 255.0);

    // 转换为 CHW 格式，直接写入 pinned 输入缓冲
    std::vector<cv::Mat> chw;
    chw.reserve(3);
    for (int c = 0; c < 3; c++) {
        chw.emplace_back(input_h_, input_w_, CV_32F, host_input_buffer_ + c * input_h_ * input_w_);
    }
    cv::split(input_float_, chw);

    // pinned 内存的异步拷贝走 DMA，不阻塞 CPU
    cudaMemcpyAsync(buffers_[0], host_input_buffer_,
                   input_size_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
}

void YOLOV5::doInference()
{
    // TensorRT 8.6+ 使用 enqueueV3
    context_->setTensorAddress(input_tensor_name_.c_str(), buffers_[0]);
    context_->setTensorAddress(output_tensor_name_.c_str(), buffers_[1]);
    context_->enqueueV3(stream_);
    
    // 拷贝输出回 CPU
    cudaMemcpyAsync(host_output_buffer_, buffers_[1],
                   output_size_ * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}
#endif

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  tmp_img_ = raw_img;
  if (use_roi_) {
    if (roi_.width == -1) {
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

#ifdef USE_TENSORRT
  // TensorRT 推理流程
  preProcess(bgr_img);
  doInference();

  // 计算缩放比例（用于还原到原图坐标）
  auto x_scale = static_cast<double>(input_h_) / bgr_img.rows;
  auto y_scale = static_cast<double>(input_w_) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  
  // 输出格式: [num_detections, 22]
  // 0-7: 4个角点 (x1,y1, x2,y2, x3,y3, x4,y4)
  // 8: 置信度
  // 9-12: 颜色 (4类)
  // 13-21: 数字 (9类)
  int output_dim = 22;  // 8 + 1 + 4 + 9 = 22
  int max_detections = output_size_ / output_dim;
  
  //tools::logger()->debug("Output: max_detections={}, output_dim={}", max_detections, output_dim);
  
  cv::Mat output(max_detections, output_dim, CV_32F, host_output_buffer_);

  return parse(scale, output, raw_img, frame_count);
#else
  return std::list<Armor>();
#endif
}

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  
  // 计算预处理时的填充偏移
  // 在preProcess中，我们将图像缩放后放在640x640画布的中心
  // 需要计算水平和垂直填充
  auto w = static_cast<int>(bgr_img.cols * scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  int pad_w = (input_w_ - w) / 2;
  int pad_h = (input_h_ - h) / 2;

  for (int r = 0; r < output.rows; r++) {
    const float* row = output.ptr<float>(r);

    // 第8列是置信度(logit)。sigmoid 单调递增，先与 logit 阈值比较，
    // 避免对 25200 行逐行计算 exp
    if (row[8] < logit_threshold_) continue;

    double score = sigmoid(row[8]);

    std::vector<cv::Point2f> armor_key_points;

    // 步骤1: 获取模型输出的原始坐标（相对于640x640输入图像）
    float x1_model = row[0];
    float y1_model = row[1];
    float x2_model = row[2];
    float y2_model = row[3];
    float x3_model = row[4];
    float y3_model = row[5];
    float x4_model = row[6];
    float y4_model = row[7];

    // 步骤2: 减去填充偏移，然后除以scale，得到在原图中的坐标
    // 注意：这里得到的是相对于当前处理图像的坐标（可能是ROI区域）
    cv::Point2f p1((x1_model - pad_w) / scale, (y1_model - pad_h) / scale);
    cv::Point2f p2((x2_model - pad_w) / scale, (y2_model - pad_h) / scale);
    cv::Point2f p3((x3_model - pad_w) / scale, (y3_model - pad_h) / scale);
    cv::Point2f p4((x4_model - pad_w) / scale, (y4_model - pad_h) / scale);
    // 如果模型输出顺序不对，在 parse 里调整
    armor_key_points.push_back(p1);  
    armor_key_points.push_back(p4);  
    armor_key_points.push_back(p3);  
    armor_key_points.push_back(p2); 

    //   // ==================== 调试绘制代码开始 ====================
    // // 创建一个用于调试显示的图像副本
    // cv::Mat debug_img = bgr_img.clone();
    
    // // 定义颜色（BGR格式）：红、绿、蓝、黄
    // std::vector<cv::Scalar> point_colors = {
    //     cv::Scalar(0, 0, 255),    // p1 - 红色 (在重排后是第4个点)
    //     cv::Scalar(0, 255, 0),    // p2 - 绿色 (在重排后是第1个点)
    //     cv::Scalar(255, 0, 0),    // p3 - 蓝色 (在重排后是第2个点)
    //     cv::Scalar(0, 255, 255)   // p4 - 黄色 (在重排后是第3个点)
    // };
    
    // // 原始顺序的点是 p1, p2, p3, p4
    // std::vector<cv::Point2f> raw_points = {p1, p2, p3, p4};
    // std::vector<std::string> raw_labels = {"p2", "p3", "p4", "p1"};
    
    // // 绘制原始模型输出的点（小圆点+标签）
    // for (int i = 0; i < 4; i++) {
    //     // 画点
    //     cv::circle(debug_img, raw_points[i], 6, point_colors[i], -1);
    //     cv::circle(debug_img, raw_points[i], 8, cv::Scalar(255, 255, 255), 2); // 白边
        
    //     // 画文字标签 (p1, p2, p3, p4)
    //     cv::putText(debug_img, raw_labels[i], 
    //                cv::Point(raw_points[i].x - 10, raw_points[i].y - 15),
    //                cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[i], 2);
        
    //     // 画序号 (0, 1, 2, 3) - 表示在armor_key_points中的索引
    //     // 注意：p2是0, p3是1, p4是2, p1是3
    //     int idx_in_vector = (i == 1) ? 0 : (i == 2) ? 1 : (i == 3) ? 2 : 3;
    //     cv::putText(debug_img, std::to_string(idx_in_vector), 
    //                cv::Point(raw_points[i].x - 5, raw_points[i].y + 25),
    //                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    // }
    
    // // 用粗线连接重排后的顺序 (p2->p3->p4->p1)
    // std::vector<cv::Point2f> ordered_points = {p2, p3, p4, p1};
    // for (int i = 0; i < 4; i++) {
    //     cv::Point2f pt1 = ordered_points[i];
    //     cv::Point2f pt2 = ordered_points[(i + 1) % 4];
    //     cv::line(debug_img, pt1, pt2, cv::Scalar(0, 255, 0), 2); // 绿色线
        
    //     // 在线的中点画箭头方向
    //     cv::Point2f mid((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
    //     cv::circle(debug_img, mid, 3, cv::Scalar(0, 0, 255), -1); // 红点标记方向
    // }
    
    // // 显示结果
    // cv::imshow("Armor Key Points Debug", debug_img);
    // cv::waitKey(1);  // 调试用，每帧暂停1ms
    // // ==================== 调试绘制代码结束 ====================


    // 颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     // 4类颜色
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  // 9类数字
    
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    
    _class_id = class_id.x;
    _color_id = color_id.x;

    // 计算包围框
    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (size_t i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  // NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      // 注意：这里armors_key_points已经包含了翻转后的坐标
      // 如果使用了ROI，还需要加上偏移
      std::vector<cv::Point2f> adjusted_points = armors_key_points[i];
      for (auto& point : adjusted_points) {
        point.x += offset_.x;
        point.y += offset_.y;
      }
      
      // 重新计算包围框
      float min_x = adjusted_points[0].x;
      float max_x = adjusted_points[0].x;
      float min_y = adjusted_points[0].y;
      float max_y = adjusted_points[0].y;
      for (size_t j = 1; j < adjusted_points.size(); j++) {
        if (adjusted_points[j].x < min_x) min_x = adjusted_points[j].x;
        if (adjusted_points[j].x > max_x) max_x = adjusted_points[j].x;
        if (adjusted_points[j].y < min_y) min_y = adjusted_points[j].y;
        if (adjusted_points[j].y > max_y) max_y = adjusted_points[j].y;
      }
      cv::Rect adjusted_rect(min_x, min_y, max_x - min_x, max_y - min_y);
      
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], adjusted_rect, adjusted_points, offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  tmp_img_ = bgr_img;
  
  // 过滤和二次矫正
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;
  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);
  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", 
      armor.confidence, 
      COLORS[armor.color], 
      ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    
    // 画4个角点
    tools::draw_points(detection, armor.points, {0, 255, 0});
    
    // 画中心点文字
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
  cv::waitKey(1);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim