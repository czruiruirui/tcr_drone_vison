#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <vector>

#include "tools/logger.hpp"

namespace auto_aim
{
namespace multithread
{

// TensorRT Logger
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            tools::logger()->warn("[TensorRT] {}", msg);
        }
    }
};

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

#ifdef USE_TENSORRT
  // 加载 TensorRT engine
  std::ifstream file(model_path, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("Failed to open TensorRT engine: " + model_path);
  }
  
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  
  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();
  
  static TensorRTLogger logger;
  runtime_ = nvinfer1::createInferRuntime(logger);
  if (!runtime_) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }
  
  // 新版 API：只需 2 个参数
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize TensorRT engine");
  }
  
  // 使用固定尺寸（640x640, 3通道）
  input_size_ = 1 * 3 * 640 * 640;
  output_size_ = 25200 * 22;  // 根据 YOLO 模型调整
  
  // 预创建多个执行上下文（用于异步推理）
  int num_contexts = 4;
  for (int i = 0; i < num_contexts; i++) {
    TensorRTContext ctx;
    ctx.context = engine_->createExecutionContext();
    cudaMalloc(&ctx.input_buffer, input_size_ * sizeof(float));
    cudaMalloc(&ctx.output_buffer, output_size_ * sizeof(float));
    cudaStreamCreate(&ctx.stream);
    ctx.host_output = new float[output_size_];
    ctx.scale = 1.0;
    contexts_.push_back(ctx);
  }
  
  tools::logger()->info("[MultiThreadDetector] TensorRT initialized with {} contexts", num_contexts);
#else
  throw std::runtime_error("TensorRT not available");
#endif
}

MultiThreadDetector::~MultiThreadDetector()
{
#ifdef USE_TENSORRT
  for (auto& ctx : contexts_) {
    // 新版 API：使用 delete
    delete ctx.context;
    if (ctx.input_buffer) cudaFree(ctx.input_buffer);
    if (ctx.output_buffer) cudaFree(ctx.output_buffer);
    if (ctx.stream) cudaStreamDestroy(ctx.stream);
    delete[] ctx.host_output;
  }
  // 新版 API：使用 delete
  delete engine_;
  delete runtime_;
#endif
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
#ifdef USE_TENSORRT
  // 获取一个空闲的上下文（简单轮询）
  static size_t ctx_idx = 0;
  size_t idx = ctx_idx % contexts_.size();
  ctx_idx++;
  
  TensorRTContext& ctx = contexts_[idx];
  
  // 计算缩放比例
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);
  ctx.scale = scale;

  // 预处理
  cv::Mat input(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
  auto roi = cv::Rect((640 - w) / 2, (640 - h) / 2, w, h);
  cv::resize(img, input(roi), {w, h});
  
  // 转换为 RGB 并归一化
  cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
  input.convertTo(input, CV_32FC3, 1.0 / 255.0);
  
  // 转换为 CHW 格式
  std::vector<cv::Mat> chw(3);
  cv::split(input, chw);
  
  // 拷贝到 GPU
  float* host_input = new float[input_size_];
  for (int c = 0; c < 3; c++) {
    memcpy(host_input + c * 640 * 640, chw[c].data, 640 * 640 * sizeof(float));
  }
  
  cudaMemcpyAsync(ctx.input_buffer, host_input, 
                 input_size_ * sizeof(float),
                 cudaMemcpyHostToDevice, ctx.stream);
  
  delete[] host_input;
  
  // 新版 API：使用 executeV2 替代 enqueueV2
  void* bindings[] = {ctx.input_buffer, ctx.output_buffer};
  ctx.context->executeV2(bindings);
  
  // 将上下文索引加入队列
  queue_.push({img.clone(), t, idx});
#endif
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, t, ctx_idx] = queue_.pop();
  
#ifdef USE_TENSORRT
  TensorRTContext& ctx = contexts_[ctx_idx];
  
  // 等待该 CUDA 流完成
  cudaStreamSynchronize(ctx.stream);
  
  // 拷贝输出回 CPU
  cudaMemcpyAsync(ctx.host_output, ctx.output_buffer, 
                 output_size_ * sizeof(float),
                 cudaMemcpyDeviceToHost, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
  
  // 转换为 cv::Mat
  int output_dim = 4 + 13 + 8;  // 根据 YOLO 模型调整
  int num_detections = output_size_ / output_dim;
  cv::Mat output(num_detections, output_dim, CV_32F, ctx.host_output);
  
  // 后处理
  auto armors = yolo_.postprocess(ctx.scale, output, img, 0);
  
  return {std::move(armors), t};
#else
  return {std::list<Armor>(), t};
#endif
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, t, ctx_idx] = queue_.pop();
  
#ifdef USE_TENSORRT
  TensorRTContext& ctx = contexts_[ctx_idx];
  
  // 等待该 CUDA 流完成
  cudaStreamSynchronize(ctx.stream);
  
  // 拷贝输出回 CPU
  cudaMemcpyAsync(ctx.host_output, ctx.output_buffer, 
                 output_size_ * sizeof(float),
                 cudaMemcpyDeviceToHost, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
  
  // 转换为 cv::Mat
  int output_dim = 4 + 13 + 8;
  int num_detections = output_size_ / output_dim;
  cv::Mat output(num_detections, output_dim, CV_32F, ctx.host_output);
  
  // 后处理
  auto armors = yolo_.postprocess(ctx.scale, output, img, 0);
  
  return {img, std::move(armors), t};
#else
  return {img, std::list<Armor>(), t};
#endif
}

}  // namespace multithread
}  // namespace auto_aim