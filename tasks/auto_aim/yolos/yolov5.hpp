#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 替换 OpenVINO 为 TensorRT
#ifdef USE_TENSORRT
#include "NvInfer.h"
#include <cuda_runtime.h>
#endif

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class YOLOV5 : public YOLOBase
{
public:
  YOLOV5(const std::string & config_path, bool debug);
  ~YOLOV5();  // 添加析构函数

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;
  bool flip_about_center_;  // 新增：是否镜像翻转

  const int class_num_ = 13;
  const int input_h_ = 640;
  const int input_w_ = 640;
  const float nms_threshold_ = 0.3;
  float score_threshold_ = 0.3;  // 可在 yaml 中通过 score_threshold 配置
  double min_confidence_, binary_threshold_;
  double logit_threshold_ = 0.0;  // score_threshold_ 对应的 logit 阈值，用于 parse 预过滤

  // TensorRT 成员
#ifdef USE_TENSORRT
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  void* buffers_[2] = {nullptr, nullptr};
  cudaStream_t stream_ = nullptr;
  float* host_input_buffer_ = nullptr;   // pinned host 输入缓冲
  float* host_output_buffer_ = nullptr;  // pinned host 输出缓冲
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  cv::Mat letterbox_canvas_;  // 复用的 640x640 预处理画布(CV_8UC3)
  cv::Mat input_float_;       // 复用的归一化浮点输入(CV_32FC3)
  
  // 新增：TensorRT 8.6+ 需要的张量名称
  std::string input_tensor_name_;
  std::string output_tensor_name_;
#endif

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;
  friend class MultiThreadDetector;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);

  // TensorRT 辅助函数
#ifdef USE_TENSORRT
  bool loadEngine(const std::string& engine_path);
  void preProcess(const cv::Mat& bgr_img);
  void doInference();
#endif
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV5_HPP