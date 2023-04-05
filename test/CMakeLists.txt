cmake_minimum_required(VERSION 2.8.7)
project(vision_sample)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -pthread -O3 ")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# test
include_directories(
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/algorithm/inc
)

file(GLOB_RECURSE SRC_LIST
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/app/*.cpp
)

# link_directories(${PROJECT_SOURCE_DIR}/algorithm/libs)
link_directories(${PROJECT_SOURCE_DIR}/../algorithm/lib)

add_executable(${PROJECT_NAME}_test ${SRC_LIST})
target_link_libraries(${PROJECT_NAME}_test
    ${OpenCV_LIBS}
    trt_detector
    omt_tracker
)
