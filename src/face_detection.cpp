#include "face_detection.h"

FaceDetector::FaceDetector(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "FaceDetection"),
      session(env, model_path.c_str(), session_options) {}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(320, 240));
    resized.convertTo(resized, CV_32F, 1.0 / 255);

    std::vector<int64_t> input_dims = {1, 3, 240, 320};
    std::vector<float> input_tensor_values(resized.begin<float>(), resized.end<float>());
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::AllocatorWithDefaultOptions(), input_tensor_values.data(),
        input_tensor_values.size(), input_dims.data(), input_dims.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_tensor);
    return std::vector<cv::Rect>{}; 
}
