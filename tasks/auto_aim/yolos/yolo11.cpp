#include "yolo11.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

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

YOLO11::YOLO11(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolo11_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

#ifdef USE_TENSORRT
  // 加载 TensorRT engine
  if (!loadEngine(model_path_)) {
    throw std::runtime_error("Failed to load TensorRT engine: " + model_path_);
  }
#endif
}

#ifdef USE_TENSORRT
bool YOLO11::loadEngine(const std::string& engine_path)
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
    
    // 修复：计算输出维度（YOLO11 输出格式: 4 + 38 + 8 = 50）
    output_dim_ = output_dims.d[output_dims.nbDims - 1];  // 通常是 50
    
    tools::logger()->info("YOLO11 TensorRT Engine loaded:");
    tools::logger()->info("  Input size: {}", input_size_);
    tools::logger()->info("  Output size: {}", output_size_);
    tools::logger()->info("  Output dim per detection: {}", output_dim_);
    
    // 分配 GPU 内存
    cudaMalloc(&buffers_[0], input_size_ * sizeof(float));
    cudaMalloc(&buffers_[1], output_size_ * sizeof(float));
    cudaStreamCreate(&stream_);
    
    // 分配 CPU 内存用于输出
    host_output_buffer_ = new float[output_size_];
    
    // 保存张量名称供后续使用
    input_tensor_name_ = input_name;
    output_tensor_name_ = output_name;
    
    return true;
}

void YOLO11::preProcess(const cv::Mat& bgr_img, cv::Mat& input_blob)
{
    // 创建 640x640 的输入图像
    input_blob = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));  // 灰色填充
    
    auto x_scale = static_cast<double>(640) / bgr_img.rows;
    auto y_scale = static_cast<double>(640) / bgr_img.cols;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(bgr_img.rows * scale);
    auto w = static_cast<int>(bgr_img.cols * scale);
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(bgr_img, resized, {w, h});
    
    // 居中放置
    auto roi = cv::Rect((640 - w) / 2, (640 - h) / 2, w, h);
    resized.copyTo(input_blob(roi));
    
    // 转换为 RGB 并归一化到 [0, 1]
    cv::cvtColor(input_blob, input_blob, cv::COLOR_BGR2RGB);
    input_blob.convertTo(input_blob, CV_32FC3, 1.0 / 255.0);
    
    // 转换为 CHW 格式
    std::vector<cv::Mat> chw(3);
    cv::split(input_blob, chw);
    
    // 拷贝到 GPU
    float* host_input = new float[input_size_];
    for (int c = 0; c < 3; c++) {
        memcpy(host_input + c * 640 * 640, chw[c].data, 640 * 640 * sizeof(float));
    }
    
    cudaMemcpyAsync(buffers_[0], host_input, 
                   input_size_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
    
    delete[] host_input;
}

void YOLO11::doInference()
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

std::list<Armor> YOLO11::detect(const cv::Mat & raw_img, int frame_count)
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
  cv::Mat input_blob;
  preProcess(bgr_img, input_blob);
  doInference();
  
  // 计算缩放比例
  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  
  // 将输出转换为 cv::Mat 格式
  // YOLO11 输出格式: [num_detections, 4 + 38 + 8] = [num_detections, 50]
  int max_detections = output_size_ / output_dim_;
  cv::Mat output(max_detections, output_dim_, CV_32F, host_output_buffer_);
  
  return parse(scale, output, raw_img, frame_count);
#else
  return std::list<Armor>();
#endif
}

std::list<Armor> YOLO11::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: xywh + classes + keypoints
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  
  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, 4 + class_num_ + 8);

    std::vector<cv::Point2f> armor_key_points;

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;

    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    for (int i = 0; i < 4; i++) {
      float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }
    ids.emplace_back(max_point.x);
    confidences.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

bool YOLO11::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  return name_ok && confidence_ok;
}

bool YOLO11::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  return name_ok;
}

cv::Point2f YOLO11::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLO11::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

void YOLO11::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

void YOLO11::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

std::list<Armor> YOLO11::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim