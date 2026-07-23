#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>

#ifdef USE_TENSORRT
#include "NvInfer.h"
#include <cuda_runtime.h>
#endif

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

#ifdef USE_TENSORRT
// TensorRT 上下文结构（替代 ov::InferRequest）
struct TensorRTContext {
    nvinfer1::IExecutionContext* context = nullptr;
    void* input_buffer = nullptr;
    void* output_buffer = nullptr;
    cudaStream_t stream = nullptr;
    float* host_output = nullptr;
    double scale = 1.0;
};
#endif

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);
  ~MultiThreadDetector();

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  std::string device_;
  YOLO yolo_;

#ifdef USE_TENSORRT
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  std::vector<TensorRTContext> contexts_;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
#endif

  // 队列中存储上下文索引（替代 ov::InferRequest）
  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, size_t>>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP