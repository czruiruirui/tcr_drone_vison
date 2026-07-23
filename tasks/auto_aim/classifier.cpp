#include "classifier.hpp"

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>

#include "tools/logger.hpp"

namespace auto_aim
{

// TensorRT Logger
class ClassifierLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            tools::logger()->warn("[TensorRT Classifier] {}", msg);
        }
    }
};

Classifier::Classifier(const std::string & config_path)
: runtime_(nullptr), engine_(nullptr), context_(nullptr),
  stream_(nullptr), host_input_buffer_(nullptr), host_output_buffer_(nullptr),
  input_size_(0), output_size_(0), use_tensorrt_(false)
{
  auto yaml = YAML::LoadFile(config_path);
  auto model = yaml["classify_model"].as<std::string>();
  
#ifdef USE_TENSORRT
  // 尝试加载 TensorRT engine
  std::string engine_path = model;
  size_t pos = engine_path.find_last_of('.');
  if (pos != std::string::npos) {
    engine_path = engine_path.substr(0, pos) + ".engine";
  }
  
  if (loadEngine(engine_path)) {
    tools::logger()->info("[Classifier] TensorRT engine loaded: {}", engine_path);
    use_tensorrt_ = true;
    
    // 分配主机内存 (1x1x32x32 input, 1x9 output)
    input_size_ = 1 * 1 * 32 * 32;
    output_size_ = 1 * 9;
    host_input_buffer_ = new float[input_size_];
    host_output_buffer_ = new float[output_size_];
  } else {
    tools::logger()->warn("[Classifier] Failed to load TensorRT, using OpenCV DNN");
    net_ = cv::dnn::readNetFromONNX(model);
  }
#else
  net_ = cv::dnn::readNetFromONNX(model);
  tools::logger()->info("[Classifier] Using OpenCV DNN");
#endif
}

Classifier::~Classifier()
{
#ifdef USE_TENSORRT
  if (host_input_buffer_) {
    delete[] host_input_buffer_;
    host_input_buffer_ = nullptr;
  }
  if (host_output_buffer_) {
    delete[] host_output_buffer_;
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
bool Classifier::loadEngine(const std::string& engine_path)
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
  
  static ClassifierLogger logger;
  runtime_ = nvinfer1::createInferRuntime(logger);
  if (!runtime_) {
    tools::logger()->error("Failed to create TensorRT runtime");
    return false;
  }
  
  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
  if (!engine_) {
    tools::logger()->error("Failed to deserialize TensorRT engine");
    return false;
  }
  
  context_ = engine_->createExecutionContext();
  if (!context_) {
    tools::logger()->error("Failed to create TensorRT execution context");
    return false;
  }
  
  cudaStreamCreate(&stream_);
  
  // 分配 GPU 内存
  cudaMalloc(&buffers_[0], 1 * 1 * 32 * 32 * sizeof(float));
  cudaMalloc(&buffers_[1], 1 * 9 * sizeof(float));
  
  return true;
}
#endif

void Classifier::classify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

#ifdef USE_TENSORRT
  if (use_tensorrt_) {
    // TensorRT 推理
    cv::Mat gray;
    cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

    // Resize to 32x32，保持长宽比，左上角对齐
    cv::Mat input(32, 32, CV_8UC1, cv::Scalar(0));
    auto x_scale = static_cast<double>(32) / gray.cols;
    auto y_scale = static_cast<double>(32) / gray.rows;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(gray.rows * scale);
    auto w = static_cast<int>(gray.cols * scale);

    if (h == 0 || w == 0) {
      armor.name = ArmorName::not_armor;
      return;
    }
    
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(gray, input(roi), {w, h});
    
    // 归一化到 [0, 1]
    input.convertTo(input, CV_32F, 1.0 / 255.0);
    
    // 拷贝到 host buffer
    memcpy(host_input_buffer_, input.data, 32 * 32 * sizeof(float));
    
    // 拷贝到 GPU
    cudaMemcpyAsync(buffers_[0], host_input_buffer_, input_size_ * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
    
    // 获取张量名称并推理
    const char* input_name = engine_->getIOTensorName(0);
    const char* output_name = engine_->getIOTensorName(1);
    context_->setTensorAddress(input_name, buffers_[0]);
    context_->setTensorAddress(output_name, buffers_[1]);
    context_->enqueueV3(stream_);
    
    // 拷贝结果回 CPU
    cudaMemcpyAsync(host_output_buffer_, buffers_[1], output_size_ * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    
    // Softmax 处理
    cv::Mat outputs(1, 9, CV_32F, host_output_buffer_);
    float max_val = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max_val, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;

    double confidence;
    cv::Point label_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
    
    armor.confidence = confidence;
    armor.name = static_cast<ArmorName>(label_point.x);
    return;
  }
#endif
  
  // OpenCV DNN 回退
  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  net_.setInput(blob);
  cv::Mat outputs = net_.forward();

  // softmax
  float max_val = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max_val, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_point.x);
}

void Classifier::ovclassify(Armor & armor)
{
  // 兼容旧接口，直接调用 classify
  classify(armor);
}

}  // namespace auto_aim