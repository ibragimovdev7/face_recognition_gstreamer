#include "face_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string pipeline =
        "rtspsrc location=rtsp://admin:123456@192.168.1.112:554/stream_0 ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video stream!" << std::endl;
        return -1;
    }

    FaceDetector face_detector("models/yolov5_face.onnx");

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto faces = face_detector.detect(frame);
        for (const auto& face : faces) {
            cv::rectangle
::contentReference[oaicite:0]{index=0}
 
