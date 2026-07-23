cmake_minimum_required(VERSION 3.16)

# 删除所有 TensorRT 配置代码，直接使用顶层的 TENSORRT_LIBRARIES

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

# 使用 ${TENSORRT_LIBRARIES} 替代 openvino::runtime
target_link_libraries(auto_aim io ${TENSORRT_LIBRARIES} tinympcstatic)