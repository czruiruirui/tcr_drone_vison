cmake_minimum_required(VERSION 3.16)

find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    buff_target.cpp
    yolo11_buff.cpp
)

# 链接库：auto_aim 已经包含 TensorRT/CUDA，再显式添加 CERES
target_link_libraries(auto_buff 
    auto_aim 
    ${CERES_LIBRARIES}
)

# 确保能使用父目录定义的 USE_TENSORRT 宏
target_compile_definitions(auto_buff PRIVATE 
    $<$<BOOL:${USE_TENSORRT}>:USE_TENSORRT>
)