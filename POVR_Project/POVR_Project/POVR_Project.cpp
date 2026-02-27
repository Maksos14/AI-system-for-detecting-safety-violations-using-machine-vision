#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    // Open camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Camera error" << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frameCount = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        frameCount++;

        // Show versions on video
        cv::putText(frame, "OpenCV: " + std::string(CV_VERSION),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        cv::putText(frame, "ONNX Runtime: 1.24.2",
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Camera Demo", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}