cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

# ==================== TensorRT 配置 (替代 OpenVINO) ====================
# 查找 CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    message(STATUS "CUDA not found")
endif()

# 查找 TensorRT
find_library(TENSORRT_LIBRARY nvinfer 
    HINTS /usr/lib/aarch64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
    PATHS /usr/local/cuda/lib64 /usr/local/tensorrt/lib
)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin 
    HINTS /usr/lib/aarch64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
    PATHS /usr/local/cuda/lib64 /usr/local/tensorrt/lib
)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
    HINTS /usr/include/aarch64-linux-gnu/ /usr/include/x86_64-linux-gnu/
    PATHS /usr/local/cuda/include /usr/local/tensorrt/include
)

if(TENSORRT_LIBRARY AND TENSORRT_INCLUDE_DIR)
    message(STATUS "TensorRT found:")
    message(STATUS "  Library: ${TENSORRT_LIBRARY}")
    message(STATUS "  Plugin: ${TENSORRT_PLUGIN_LIBRARY}")
    message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
    
    include_directories(${TENSORRT_INCLUDE_DIR})
    add_definitions(-DUSE_TENSORRT)
    
    set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY})
else()
    message(WARNING "TensorRT not found! Inference functionality will be disabled.")
    set(TENSORRT_LIBRARIES "")
endif()

# 可选：查找 cuDNN
find_library(CUDNN_LIBRARY cudnn 
    HINTS /usr/lib/aarch64-linux-gnu/ /usr/local/cuda/lib64
)
if(CUDNN_LIBRARY)
    message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
    list(APPEND TENSORRT_LIBRARIES ${CUDNN_LIBRARY})
else()
    message(STATUS "cuDNN not found")
endif()

# 设置 CUDA 和 TensorRT 链接库
set(CUDA_AND_TRT_LIBS ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})
message(STATUS "CUDA and TensorRT libs: ${CUDA_AND_TRT_LIBS}")

# ==================== TensorRT 配置结束 ====================

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)

####################src###################
add_executable(standard src/standard.cpp)
add_executable(mt_standard src/mt_standard.cpp)
add_executable(standard_mpc src/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc src/auto_aim_debug_mpc.cpp)
add_executable(mt_auto_aim_debug src/mt_auto_aim_debug.cpp)
add_executable(auto_buff_debug src/auto_buff_debug.cpp)
add_executable(auto_buff_debug_mpc src/auto_buff_debug_mpc.cpp)
add_executable(uav src/uav.cpp)
add_executable(uav_debug src/uav_debug.cpp)

target_link_libraries(standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(mt_standard ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(standard_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(auto_aim_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(mt_auto_aim_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(auto_buff_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES} ${CUDA_AND_TRT_LIBS})
target_link_libraries(auto_buff_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES} ${CUDA_AND_TRT_LIBS})
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(uav_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io ${CUDA_AND_TRT_LIBS})

##################calibration################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

##################tests##################
add_executable(auto_aim_test tests/auto_aim_test.cpp)
add_executable(auto_buff_test tests/auto_buff_test.cpp)
add_executable(camera_detect_test tests/camera_detect_test.cpp)
add_executable(camera_test tests/camera_test.cpp)
add_executable(camera_thread_test tests/camera_thread_test.cpp)
add_executable(cboard_test tests/cboard_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(multi_usbcamera_test tests/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(minimum_vision_system tests/minimum_vision_system.cpp)

target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim ${CUDA_AND_TRT_LIBS})
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(camera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim io ${CUDA_AND_TRT_LIBS})
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_thread_test ${OpenCV_LIBS} fmt::fmt auto_aim tools io ${CUDA_AND_TRT_LIBS})
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim ${CUDA_AND_TRT_LIBS})
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(usbcamera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim ${CUDA_AND_TRT_LIBS})
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim ${CUDA_AND_TRT_LIBS})
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(minimum_vision_system ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim ${CUDA_AND_TRT_LIBS})

add_executable(gimbal_test tests/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io ${CUDA_AND_TRT_LIBS})

add_executable(planner_test tests/planner_test.cpp)
target_link_libraries(planner_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io ${CUDA_AND_TRT_LIBS})

add_executable(planner_test_offline tests/planner_test_offline.cpp)
target_link_libraries(planner_test_offline ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io ${CUDA_AND_TRT_LIBS})

# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
    # 编译 sentry 和 publish_test
    add_executable(sentry src/sentry.cpp)
    add_executable(sentry_bp src/sentry_bp.cpp)
    add_executable(sentry_debug src/sentry_debug.cpp)
    add_executable(sentry_multithread src/sentry_multithread.cpp)
    add_executable(publish_test tests/publish_test.cpp)
    add_executable(subscribe_test tests/subscribe_test.cpp)
    add_executable(topic_loop_test tests/topic_loop_test.cpp)

    target_link_libraries(sentry ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io ${CUDA_AND_TRT_LIBS})
    target_link_libraries(sentry_bp ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io ${CUDA_AND_TRT_LIBS})
    target_link_libraries(sentry_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io ${CUDA_AND_TRT_LIBS})
    target_link_libraries(sentry_multithread ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim omniperception tools io ${CUDA_AND_TRT_LIBS})
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim ${CUDA_AND_TRT_LIBS})
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()