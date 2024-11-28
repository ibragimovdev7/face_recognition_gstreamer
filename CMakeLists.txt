cmake_minimum_required(VERSION 3.11)
project(FaceRecognitionGStreamer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)

include_directories(${OpenCV_INCLUDE_DIRS} ${GST_INCLUDE_DIRS} include)
link_directories(${GST_LIBRARY_DIRS})

add_executable(FaceRecognitionGStreamer src/main.cpp)
target_link_libraries(FaceRecognitionGStreamer ${OpenCV_LIBS} ${GST_LIBRARIES})
