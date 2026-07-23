#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <string>

#ifdef USE_TENSORRT
#include "NvInfer.h"
#include <cuda_runtime.h>
#endif

#include "armor.hpp"

namespace auto_aim
{
class Classifier
{
public:
  explicit Classifier(const std::string & config_path);
  ~Classifier();

  void classify(Armor & armor);
  
  // 保留兼容旧接口
  void ovclassify(Armor & armor);

private:
  cv::dnn::Net net_;

#ifdef USE_TENSORRT
  // TensorRT 成员
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  void* buffers_[2] = {nullptr, nullptr};
  cudaStream_t stream_ = nullptr;
  
  float* host_input_buffer_ = nullptr;
  float* host_output_buffer_ = nullptr;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  
  bool use_tensorrt_ = false;
  
  // 加载引擎
  bool loadEngine(const std::string& engine_path);
#endif
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP