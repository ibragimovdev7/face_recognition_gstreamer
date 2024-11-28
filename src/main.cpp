#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_detection.h"

int main(int argc, char* argv[]) {
    std::string pipeline = 
        "rtspsrc location=rtsp://admin:123456@192.168.1.112:554/stream_0 ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video stream!" << std::endl;
        return -1;
    }

    FaceDetector face_detector("models/ultralight_face_detector.onnx");

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        std::vector<cv::Rect> faces = face_detector.detect(frame);

        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Face Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    return 0;
}
