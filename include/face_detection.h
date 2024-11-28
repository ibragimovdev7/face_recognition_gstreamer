#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

class FaceDetector {
public:
    explicit FaceDetector(const std::string& model_path);
    std::vector<cv::Rect> detect(const cv::Mat& frame);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    size_t input_tensor_size;
    std::vector<int64_t> input_node_dims;
    std::vector<float> input_image_;
    const float confidence_threshold = 0.5f;
    const float nms_threshold = 0.4f;

    std::vector<cv::Rect> non_max_suppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences);
};

#endif // FACE_DETECTOR_H
