#include "face_detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>

FaceDetector::FaceDetector(const std::string& model_path) {
    if (!face_cascade.load(model_path)) {
        throw std::runtime_error("Yuzni aniqlash modeli yuklanmadi: " + model_path);
    }
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(50, 50));
    return faces;
}

int main(int argc, char* argv[]) {
    std::string pipeline =
        "rtspsrc location=rtsp://rtsp://admin:123456@192.168.1.112:554/stream_0 ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video stream!" << std::endl;
        return -1;
    }

    FaceDetector face_detector("models/haarcascade_frontalface_default.xml");

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Yuzlarni aniqlash
        std::vector<cv::Rect> faces = face_detector.detect(frame);

        // Aniqlangan yuzlarni ramka bilan belgilash
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
        }

        // Natijani ko‘rsatish
        cv::imshow("Face Detection", frame);

        // Chiqish uchun 'q' tugmasini bosing
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
