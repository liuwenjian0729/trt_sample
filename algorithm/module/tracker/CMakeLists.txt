cmake_minimum_required(VERSION 2.8.7)
project(omt_tracker)

set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/omt_tracker/include
    ${PROJECT_SOURCE_DIR}/omt_tracker/byte_tracker/include
    ${EIGEN3_INCLUDE_DIR}
)

file(GLOB BYTE_TRACKER_SRC
    "${PROJECT_SOURCE_DIR}/common/*.cc"
    "${PROJECT_SOURCE_DIR}/omt_tracker/byte_tracker/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/omt_tracker/src/*.cpp"
)

add_library(${PROJECT_NAME} SHARED ${BYTE_TRACKER_SRC})
