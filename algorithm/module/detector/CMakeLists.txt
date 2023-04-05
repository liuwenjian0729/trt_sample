cmake_minimum_required(VERSION 2.8.7)
project(trt_detector)

set(CMAKE_BUILD_TYPE "Release")

include_directories(
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/yolo
)

file(GLOB_RECURSE SRC_LIST
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# tensorrt & cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
include_directories(/usr/local/TensorRT-8.2.5.1/include)
link_directories(/usr/local/TensorRT-8.2.5.1/lib)

# yolov5
file(GLOB_RECURSE CU_LIST
    ${PROJECT_SOURCE_DIR}/yolo/*.cu
)
cuda_add_library(yolov5plugin SHARED ${CU_LIST})
target_link_libraries(yolov5plugin
    nvinfer
    cudart
)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    nvinfer
    cudart
    yolov5plugin
    -pthread
)
