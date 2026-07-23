#ifndef AUTO_AIM__YOLOV8_HPP
#define AUTO_AIM__YOLOV8_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef USE_TENSORRT
#include "NvInfer.h"
#include <cuda_runtime.h>
#endif

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/classifier.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{

class YOLOV8 : public YOLOBase
{
public:
  YOLOV8(const std::string & config_path, bool debug);
  ~YOLOV8();  // 添加析构函数

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  Classifier classifier_;
  Detector detector_;

  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;

  const int class_num_ = 2;  // YOLOv8 有 2 个类别
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

#ifdef USE_TENSORRT
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  void* buffers_[2] = {nullptr, nullptr};
  cudaStream_t stream_ = nullptr;
  size_t input_size_ = 0;
  size_t output_size_ = 0;
  float* host_output_buffer_ = nullptr;
  
  // 新增：TensorRT 8.6+ 需要的张量名称
  std::string input_tensor_name_;
  std::string output_tensor_name_;
#endif

  cv::Rect roi_;
  cv::Point2f offset_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;
  ArmorType get_type(const Armor & armor);
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);

#ifdef USE_TENSORRT
  bool loadEngine(const std::string& engine_path);
  void preProcess(const cv::Mat& bgr_img, cv::Mat& input_blob);
  void doInference();
#endif
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOV8_HPP