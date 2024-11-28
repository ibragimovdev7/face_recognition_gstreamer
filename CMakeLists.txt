cmake_minimum_required(VERSION 3.10)
project(FaceRecognitionGStreamer)

# Standartni belgilash
set(CMAKE_CXX_STANDARD 17)

# OpenCV kutubxonasini topish
find_package(OpenCV REQUIRED)

# ONNX Runtime kutubxonasini topish
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNX REQUIRED onnxruntime)

# Kataloglarni ulash
include_directories(${OpenCV_INCLUDE_DIRS} include)
link_directories(${OpenCV_LIB_DIRS})
link_directories(${ONNX_LIBRARY_DIRS})

# Yig'iladigan faylni ko'rsatish
add_executable(face_recognition_gstreamer
    src/main.cpp
    src/face_detector.cpp
    include/face_detector.h
)

# Kutubxonalarni ulash
target_link_libraries(face_recognition_gstreamer
    ${OpenCV_LIBS}
    ${ONNX_LIBRARIES}
)
