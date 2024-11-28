#include "face_detection.h"
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <iostream>

FaceDetector::FaceDetector(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "MTCNN"),
      session(env, model_path.c_str(), session_options) {
    session_options.SetIntraOpNumThreads(1);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 1));
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    
    // Tasvirni ONNX modeliga tayyorlash
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(160, 160)); // MTCNN modelining kirish hajmi
    resized.convertTo(resized, CV_32F, 1 / 255.0);
    resized = (resized - 0.5) / 0.5;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), resized.ptr<float>(),
        resized.total(), {1, 3, 160, 160});

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, &"input", &input_tensor, 1, &"output", 1);

    // Natijalarni o'qish va yuz koordinatalarini qaytarish
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    for (int i = 0; i < 10; ++i) { // Modelning chiqish o'lchamiga moslashtiring
        faces.emplace_back(cv::Rect(output_data[4 * i], output_data[4 * i + 1],
                                    output_data[4 * i + 2], output_data[4 * i + 3]));
    }

    return faces;
}

int main() {
    std::string pipeline =
        "rtspsrc location=rtsp://admin:123456@192.168.1.112:554/stream_0 ! "
        "decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video stream!" << std::endl;
        return -1;
    }

    FaceDetector face_detector("models/mtcnn.onnx");

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

        cv::imshow("RTSP Stream with Face Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
