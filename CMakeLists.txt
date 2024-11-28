cmake_minimum_required(VERSION 3.11)
project(FaceDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FaceDetection src/main.cpp src/face_detection.cpp)
target_link_libraries(FaceDetection ${OpenCV_LIBS} onnxruntime)
