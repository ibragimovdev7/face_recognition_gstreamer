#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <opencv2/opencv.hpp>

class FaceDetector {
public:
    explicit FaceDetector(const std::string& model_path);
    std::vector<cv::Rect> detect(const cv::Mat& frame);

private:
    cv::CascadeClassifier face_cascade;
};

#endif // FACE_DETECTION_H
