// 诊断工具：加载 TensorRT engine + 一张图片，打印输出张量的逐列统计和 top 行
// 用于对比 0526 / 0708 模型输出格式是否一致
// 用法: ./build/engine_dump_test <engine路径> <图片路径>

#include <fmt/core.h>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class TensorRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
  }
};

int main(int argc, char * argv[])
{
  if (argc < 3) {
    fmt::print("用法: {} <engine路径> <图片路径>\n", argv[0]);
    return 1;
  }
  const std::string engine_path = argv[1];
  const std::string img_path = argv[2];

  // 读取 engine
  std::ifstream file(engine_path, std::ios::binary);
  if (!file) {
    fmt::print(stderr, "无法打开 engine: {}\n", engine_path);
    return 1;
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  TensorRTLogger trt_logger;
  auto runtime = nvinfer1::createInferRuntime(trt_logger);
  auto engine = runtime->deserializeCudaEngine(engine_data.data(), size);
  if (!engine) {
    fmt::print(stderr, "engine 反序列化失败\n");
    return 1;
  }
  auto context = engine->createExecutionContext();

  const char * input_name = engine->getIOTensorName(0);
  const char * output_name = engine->getIOTensorName(1);
  auto input_dims = engine->getTensorShape(input_name);
  auto output_dims = engine->getTensorShape(output_name);

  fmt::print("输入维度: ");
  for (int i = 0; i < input_dims.nbDims; i++) fmt::print("{} ", input_dims.d[i]);
  fmt::print("\n输出维度: ");
  for (int i = 0; i < output_dims.nbDims; i++) fmt::print("{} ", output_dims.d[i]);
  fmt::print("\n");

  const int input_h = input_dims.d[2], input_w = input_dims.d[3];
  const int num_rows = output_dims.d[1], num_cols = output_dims.d[2];
  size_t input_size = 1ull * 3 * input_h * input_w;
  size_t output_size = 1ull * num_rows * num_cols;

  // 读取图片并做与 YOLOV5::preProcess 相同的 letterbox 预处理
  cv::Mat img = cv::imread(img_path);
  if (img.empty()) {
    fmt::print(stderr, "无法读取图片: {}\n", img_path);
    return 1;
  }
  fmt::print("图片尺寸: {}x{}\n", img.cols, img.rows);

  cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  double scale = std::min(
    static_cast<double>(input_h) / img.rows, static_cast<double>(input_w) / img.cols);
  int h = img.rows * scale, w = img.cols * scale;
  cv::Mat resized;
  cv::resize(img, resized, {w, h});
  resized.copyTo(canvas(cv::Rect((input_w - w) / 2, (input_h - h) / 2, w, h)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  cv::Mat float_img;
  canvas.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

  std::vector<float> host_input(input_size);
  std::vector<cv::Mat> chw;
  for (int c = 0; c < 3; c++)
    chw.emplace_back(input_h, input_w, CV_32F, host_input.data() + c * input_h * input_w);
  cv::split(float_img, chw);

  // 推理
  void *buffers[2];
  cudaMalloc(&buffers[0], input_size * sizeof(float));
  cudaMalloc(&buffers[1], output_size * sizeof(float));
  cudaMemcpy(buffers[0], host_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

  context->setTensorAddress(input_name, buffers[0]);
  context->setTensorAddress(output_name, buffers[1]);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context->enqueueV3(stream);
  cudaStreamSynchronize(stream);

  std::vector<float> output(output_size);
  cudaMemcpy(output.data(), buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);

  // 检查 NaN/Inf
  int nan_count = 0, inf_count = 0;
  for (auto v : output) {
    if (std::isnan(v)) nan_count++;
    if (std::isinf(v)) inf_count++;
  }
  fmt::print("输出共 {} 个值, NaN: {}, Inf: {}\n", output_size, nan_count, inf_count);

  // 逐列统计
  fmt::print("\n逐列统计 (共 {} 行 x {} 列):\n", num_rows, num_cols);
  fmt::print("{:>4} {:>12} {:>12} {:>12}\n", "列", "min", "max", "mean");
  for (int c = 0; c < num_cols; c++) {
    float vmin = 1e30, vmax = -1e30;
    double vsum = 0;
    for (int r = 0; r < num_rows; r++) {
      float v = output[r * num_cols + c];
      if (std::isnan(v)) continue;
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
      vsum += v;
    }
    fmt::print("{:>4} {:>12.4f} {:>12.4f} {:>12.4f}\n", c, vmin, vmax, vsum / num_rows);
  }

  // 按第 8 列(置信度)排序, 打印 top 5 行
  std::vector<int> idx(num_rows);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) {
    float va = output[a * num_cols + 8], vb = output[b * num_cols + 8];
    if (std::isnan(va)) return false;
    if (std::isnan(vb)) return true;
    return va > vb;
  });
  fmt::print("\n按第 8 列排序的 top 5 行原始值:\n");
  for (int i = 0; i < 5; i++) {
    int r = idx[i];
    fmt::print("行 {}: ", r);
    for (int c = 0; c < num_cols; c++) fmt::print("{:.3f} ", output[r * num_cols + c]);
    fmt::print("\n");
  }

  cudaFree(buffers[0]);
  cudaFree(buffers[1]);
  cudaStreamDestroy(stream);
  delete context;
  delete engine;
  delete runtime;
  return 0;
}
