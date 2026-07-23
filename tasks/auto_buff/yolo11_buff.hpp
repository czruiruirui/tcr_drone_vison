#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#endif

#include "tools/logger.hpp"

namespace auto_buff
{
const std::vector<std::string> class_names = {"buff", "r"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<cv::Point2f> kpt;
  };

  explicit YOLO11_BUFF(const std::string & config);
  ~YOLO11_BUFF();

  YOLO11_BUFF(const YOLO11_BUFF&) = delete;
  YOLO11_BUFF& operator=(const YOLO11_BUFF&) = delete;

  std::vector<Object> get_multicandidateboxes(cv::Mat & image);
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
#ifdef USE_TENSORRT
  // TensorRT Logger 内部类定义
  class TensorRTLogger : public nvinfer1::ILogger {
  public:
      void log(Severity severity, const char* msg) noexcept override {
          if (severity <= Severity::kWARNING) {
              std::cout << "[TensorRT] " << msg << std::endl;
          }
      }
  };
  
  // 获取单例 logger 的静态方法
  static TensorRTLogger& getLogger();

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  void* buffers_[2] = {nullptr, nullptr};
  float* host_output_buffer_ = nullptr;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  int output_dim_ = 0;
  std::string input_tensor_name_;
  std::string output_tensor_name_;

  bool loadEngine(const std::string& engine_path);
  void preProcess(const cv::Mat& input_image);
  void doInference();
#endif

  static constexpr int NUM_POINTS = 6;
  static constexpr float CONFIDENCE_THRESHOLD = 0.7f;
  static constexpr float IOU_THRESHOLD = 0.4f;
  
  int input_w_ = 640;
  int input_h_ = 640;

  float fill_tensor_data_image(const cv::Mat & input_image);
  void convert(const cv::Mat & input, cv::Mat & output, 
               const bool normalize, const bool exchangeRB) const;
  void save(const std::string & programName, const cv::Mat & image);
};

}  // namespace auto_buff
#endif