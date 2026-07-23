cmake_minimum_required(VERSION 3.16)

# 删除所有 TensorRT 配置代码

add_library(omniperception OBJECT 
    decider.cpp
    perceptron.cpp
)

# 使用 ${TENSORRT_LIBRARIES} 替代 openvino::runtime
target_link_libraries(omniperception ${TENSORRT_LIBRARIES})