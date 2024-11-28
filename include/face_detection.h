#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class FaceDetector {
public:
    explicit FaceDetector(const std::string& model_path);
    std::vector<cv::Rect> detect(const cv::Mat& frame);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
};

#endif // FACE_DETECTION_H
