#include "yolo11_buff.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>

#include "tools/logger.hpp"

namespace auto_buff
{

// 静态 logger 实现
#ifdef USE_TENSORRT
YOLO11_BUFF::TensorRTLogger& YOLO11_BUFF::getLogger() {
    static TensorRTLogger instance;
    return instance;
}
#endif

YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
    auto yaml = YAML::LoadFile(config);
    std::string model_path = yaml["model"].as<std::string>();
    
#ifdef USE_TENSORRT
    if (!loadEngine(model_path)) {
        throw std::runtime_error("Failed to load TensorRT engine: " + model_path);
    }
#endif
}

YOLO11_BUFF::~YOLO11_BUFF()
{
#ifdef USE_TENSORRT
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
bool YOLO11_BUFF::loadEngine(const std::string& engine_path)
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
    
    // 使用 getLogger() 替代局部的 static 变量
    runtime_ = nvinfer1::createInferRuntime(getLogger());
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
    
    output_dim_ = output_dims.d[output_dims.nbDims - 1];
    
    tools::logger()->info("YOLO11_BUFF TensorRT Engine loaded:");
    tools::logger()->info("  Input size: {}", input_size_);
    tools::logger()->info("  Output size: {}", output_size_);
    tools::logger()->info("  Output dim: {}", output_dim_);
    
    cudaMalloc(&buffers_[0], input_size_ * sizeof(float));
    cudaMalloc(&buffers_[1], output_size_ * sizeof(float));
    cudaStreamCreate(&stream_);
    
    host_output_buffer_ = new float[output_size_];
    
    input_tensor_name_ = input_name;
    output_tensor_name_ = output_name;
    
    return true;
}

void YOLO11_BUFF::preProcess(const cv::Mat& input_image)
{
    cv::Mat blob;
    cv::Mat resized;
    
    float scale = std::min(input_w_ / (float)input_image.cols, input_h_ / (float)input_image.rows);
    int new_w = static_cast<int>(input_image.cols * scale);
    int new_h = static_cast<int>(input_image.rows * scale);
    
    cv::resize(input_image, resized, cv::Size(new_w, new_h));
    
    blob = cv::Mat(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(blob(cv::Rect((input_w_ - new_w) / 2, (input_h_ - new_h) / 2, new_w, new_h)));
    
    cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
    blob.convertTo(blob, CV_32FC3, 1.0 / 255.0);
    
    std::vector<cv::Mat> chw(3);
    cv::split(blob, chw);
    
    // 修复：使用 vector 替代裸指针，避免内存泄漏
    std::vector<float> host_input(input_size_);
    for (int c = 0; c < 3; c++) {
        memcpy(host_input.data() + c * input_h_ * input_w_, 
               chw[c].data, 
               input_h_ * input_w_ * sizeof(float));
    }
    
    cudaMemcpyAsync(buffers_[0], host_input.data(), 
                   input_size_ * sizeof(float), 
                   cudaMemcpyHostToDevice, stream_);
}

void YOLO11_BUFF::doInference()
{
    context_->setTensorAddress(input_tensor_name_.c_str(), buffers_[0]);
    context_->setTensorAddress(output_tensor_name_.c_str(), buffers_[1]);
    context_->enqueueV3(stream_);
    
    cudaMemcpyAsync(host_output_buffer_, buffers_[1], output_size_ * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}
#endif

float YOLO11_BUFF::fill_tensor_data_image(const cv::Mat & input_image)
{
#ifdef USE_TENSORRT
    preProcess(input_image);
    return std::min(input_w_ / (float)input_image.cols, input_h_ / (float)input_image.rows);
#else
    return 1.0f;
#endif
}

void YOLO11_BUFF::convert(
    const cv::Mat & input, cv::Mat & output, const bool normalize, const bool exchangeRB) const
{
    if (exchangeRB) {
        cv::cvtColor(input, output, cv::COLOR_BGR2RGB);
    } else {
        output = input.clone();
    }
    
    if (normalize) {
        output.convertTo(output, CV_32FC3, 1.0 / 255.0);
    }
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
    const int64 start = cv::getTickCount();
    std::vector<Object> objects;
    
#ifdef USE_TENSORRT
    if (image.empty()) {
        tools::logger()->warn("Empty img!, camera drop!");
        return objects;
    }

    float factor = fill_tensor_data_image(image);
    doInference();
    
    int max_detections = output_size_ / output_dim_;
    cv::Mat det_output(max_detections, output_dim_, CV_32F, host_output_buffer_);
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<float>> objects_keypoints;
    
    // 修复：遍历 rows 而不是 cols（假设输出是 [num_detections, 17]）
    for (int i = 0; i < det_output.rows; ++i) {
        float* row = det_output.ptr<float>(i);
        
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];
        float obj_conf = row[4];
        
        if (obj_conf < CONFIDENCE_THRESHOLD) continue;
        
        // 坐标转换（注意：这里假设 factor 是正确的缩放因子）
        int left = static_cast<int>((x - 0.5 * w) * factor);
        int top = static_cast<int>((y - 0.5 * h) * factor);
        int width = static_cast<int>(w * factor);
        int height = static_cast<int>(h * factor);
        
        // 解析关键点
        std::vector<float> keypoints;
        int kpt_start = 5;
        for (int k = 0; k < NUM_POINTS; ++k) {
            float kx = row[kpt_start + k * 2] * factor;
            float ky = row[kpt_start + k * 2 + 1] * factor;
            keypoints.push_back(kx);
            keypoints.push_back(ky);
        }
        
        boxes.push_back(cv::Rect(left, top, width, height));
        confidences.push_back(obj_conf);
        objects_keypoints.push_back(keypoints);
    }
    
    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, indexes);
    
    for (size_t i = 0; i < indexes.size(); ++i) {
        const int index = indexes[i];
        Object obj;
        obj.rect = boxes[index];
        obj.prob = confidences[index];
        
        const std::vector<float> & keypoint = objects_keypoints[index];
        for (int k = 0; k < NUM_POINTS; ++k) {
            float x_coord = keypoint[k * 2];
            float y_coord = keypoint[k * 2 + 1];
            obj.kpt.push_back(cv::Point2f(x_coord, y_coord));
        }
        objects.push_back(obj);
        
        // 绘制
        cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
        const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        const cv::Rect textBox(obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
        cv::putText(image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        
        const int radius = 2;
        for (int k = 0; k < NUM_POINTS; ++k) {
            cv::circle(image, obj.kpt[k], radius, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
        }
    }
    
    // 计算FPS
    const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), 
               cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
#endif
    
    return objects;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
    const int64 start = cv::getTickCount();
    std::vector<Object> objects;
    
#ifdef USE_TENSORRT
    if (image.empty()) {
        tools::logger()->warn("Empty img!, camera drop!");
        return objects;
    }

    float factor = fill_tensor_data_image(image);
    doInference();
    
    int max_detections = output_size_ / output_dim_;
    cv::Mat det_output(max_detections, output_dim_, CV_32F, host_output_buffer_);
    
    // 寻找置信度最大的框
    int best_index = -1;
    float max_confidence = 0.0f;
    
    // 修复：遍历 rows 而不是 cols
    for (int i = 0; i < det_output.rows; ++i) {
        float* row = det_output.ptr<float>(i);
        float confidence = row[4];
        if (confidence > max_confidence) {
            max_confidence = confidence;
            best_index = i;
        }
    }
    
    if (max_confidence > CONFIDENCE_THRESHOLD && best_index >= 0) {
        float* row = det_output.ptr<float>(best_index);
        
        Object obj;
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];
        
        obj.rect.x = static_cast<int>((x - 0.5 * w) * factor);
        obj.rect.y = static_cast<int>((y - 0.5 * h) * factor);
        obj.rect.width = static_cast<int>(w * factor);
        obj.rect.height = static_cast<int>(h * factor);
        obj.prob = max_confidence;
        
        // 解析关键点
        int kpt_start = 5;
        for (int k = 0; k < NUM_POINTS; ++k) {
            float kx = row[kpt_start + k * 2] * factor;
            float ky = row[kpt_start + k * 2 + 1] * factor;
            obj.kpt.push_back(cv::Point2f(kx, ky));
        }
        objects.push_back(obj);
        
        // 0.3-0.7 save
        if (max_confidence < 0.7) save(std::to_string(start), image);
        
        // 绘制
        cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
        const std::string label = "buff:" + std::to_string(max_confidence).substr(0, 4);
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        const cv::Rect textBox(obj.rect.tl().x, obj.rect.tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(image, textBox, cv::Scalar(0, 255, 255), cv::FILLED);
        cv::putText(image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        
        const int radius = 2;
        for (int k = 0; k < NUM_POINTS; ++k) {
            cv::circle(image, obj.kpt[k], radius, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
            cv::putText(image, std::to_string(k + 1), obj.kpt[k] + cv::Point2f(5, -5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
    }
    
    // 计算FPS
    const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), 
               cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
#endif
    
    return objects;
}

void YOLO11_BUFF::save(const std::string & programName, const cv::Mat & image)
{
    const std::filesystem::path saveDir = "../result/";
    if (!std::filesystem::exists(saveDir)) {
        std::filesystem::create_directories(saveDir);
    }
    const std::filesystem::path savePath = saveDir / (programName + ".jpg");
    cv::imwrite(savePath.string(), image);
}

}  // namespace auto_buff