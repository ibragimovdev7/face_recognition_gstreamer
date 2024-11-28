#include "face_detector.h"

FaceDetector::FaceDetector(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector"),
      session(env, model_path.c_str(), session_options) {
    input_node_names = {session.GetInputName(0, Ort::AllocatorWithDefaultOptions())};
    output_node_names = {session.GetOutputName(0, Ort::AllocatorWithDefaultOptions())};
    input_node_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_tensor_size = 1;
    for (auto dim : input_node_dims) {
        input_tensor_size *= dim;
    }
    input_image_.resize(input_tensor_size);
}

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_node_dims[2], input_node_dims[3]));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::memcpy(input_image_.data(), resized.data, input_tensor_size * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (size_t i = 0; i < output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount(); i += 6) {
        float confidence = output_data[i + 4];
        if (confidence > confidence_threshold) {
            int x = static_cast<int>(output_data[i] * frame.cols);
            int y = static_cast<int>(output_data[i + 1] * frame.rows);
            int w = static_cast<int>(output_data[i + 2] * frame.cols - x);
            int h = static_cast<int>(output_data[i + 3] * frame.rows - y);
            boxes.emplace_back(cv::Rect(x, y, w, h));
            confidences.push_back(confidence);
        }
    }

    return non_max_suppression(boxes, confidences);
}

std::vector<cv::Rect> FaceDetector::non_max_suppression(const std::vector<cv::Rect>& boxes, const std::vector<float>& confidences) {
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);
    std::vector<cv::Rect> result;
    for (int idx : indices) {
        result.push_back(boxes[idx]);
    }
    return result;
}
