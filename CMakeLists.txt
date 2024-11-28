cmake_minimum_required(VERSION 3.10)
project(MTCNN_Face_Detection)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)
link_directories(${OpenCV_LIB_DIRS})

add_executable(mtcnn_face_detection src/main.cpp)
target_link_libraries(mtcnn_face_detection ${OpenCV_LIBS} onnxruntime)
