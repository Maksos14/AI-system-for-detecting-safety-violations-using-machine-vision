#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <windows.h>
#include <algorithm>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    std::cout << "=== YOLOv8 DETECTION (FIXED) ===\n\n";

    // Load classes
    std::vector<std::string> classes;
    std::ifstream file("models/coco.names");
    if (!file.is_open()) {
        std::cout << "ERROR: Cannot open coco.names" << std::endl;
        system("pause");
        return -1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            classes.push_back(line);
        }
    }
    std::cout << "Classes loaded: " << classes.size() << std::endl;

    // Load model
    std::cout << "\nLoading model..." << std::endl;
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX("models/yolov8n.onnx");
        std::cout << "Model loaded successfully!" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cout << "Model loading error: " << e.what() << std::endl;
        system("pause");
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Using CPU mode" << std::endl;

    // Open camera
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap.isOpened()) {
        std::cout << "ERROR: Cannot open camera!" << std::endl;
        system("pause");
        return -1;
    }
    std::cout << "\nCamera opened! Press 'q' to exit.\n" << std::endl;

    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Create blob
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0,
            cv::Size(640, 640),
            cv::Scalar(0, 0, 0),
            true, false);

        // Run detection
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        std::vector<Detection> detections;
        float confThreshold = 0.5f;

        // Process YOLOv8 output
        for (const auto& output : outputs) {
            // Транспонируем: из [1, 84, 8400] в [8400, 84]
            cv::Mat transposed = output.reshape(1, output.size[1]).t();

            const float* data = (float*)transposed.data;
            int numBoxes = transposed.rows;
            int dimensions = transposed.cols;

            for (int i = 0; i < numBoxes; i++) {
                const float* detection = data + i * dimensions;

                // YOLOv8 формат: [x, y, w, h, class1, class2, ...]
                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                // Находим класс с максимальной уверенностью
                float maxConf = 0;
                int maxClassId = -1;
                for (int j = 4; j < dimensions; j++) {
                    float conf = detection[j];
                    if (conf > maxConf) {
                        maxConf = conf;
                        maxClassId = j - 4;
                    }
                }

                if (maxConf > confThreshold) {
                    // Конвертируем в пиксельные координаты
                    // x, y - это центр бокса
                    int boxX = (int)((x - w / 2) * frame.cols);
                    int boxY = (int)((y - h / 2) * frame.rows);
                    int boxWidth = (int)(w * frame.cols);
                    int boxHeight = (int)(h * frame.rows);

                    // Проверяем границы
                    boxX = (std::max)(0, boxX);
                    boxY = (std::max)(0, boxY);
                    boxWidth = (std::min)(boxWidth, frame.cols - boxX);
                    boxHeight = (std::min)(boxHeight, frame.rows - boxY);

                    if (boxWidth > 0 && boxHeight > 0) {
                        Detection det;
                        det.classId = maxClassId;
                        det.confidence = maxConf;
                        det.box = cv::Rect(boxX, boxY, boxWidth, boxHeight);
                        detections.push_back(det);

                        // Отладка для первых нескольких объектов
                        static int debugCount = 0;
                        if (debugCount < 5) {
                            std::cout << "Detected: class=" << maxClassId
                                << " (" << classes[maxClassId] << ")"
                                << ", conf=" << maxConf
                                << ", box=[" << boxX << "," << boxY
                                << "," << boxWidth << "x" << boxHeight << "]" << std::endl;
                            debugCount++;
                        }
                    }
                }
            }
        }

        // Non-Maximum Suppression
        std::vector<int> indices;
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;

        for (const auto& det : detections) {
            boxes.push_back(det.box);
            confs.push_back(det.confidence);
        }

        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confs, confThreshold, 0.4f, indices);
        }

        // Draw results
        for (int idx : indices) {
            const auto& det = detections[idx];

            // Цвет: зеленый для человека (classId=0), синий для остального
            cv::Scalar color = (det.classId == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);

            cv::rectangle(frame, det.box, color, 2);

            std::string label;
            if (det.classId >= 0 && det.classId < classes.size()) {
                label = classes[det.classId] + ": " + std::to_string(det.confidence).substr(0, 4);
            }
            else {
                label = "Unknown: " + std::to_string(det.confidence).substr(0, 4);
            }

            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5f, color, 1);
        }

        // Show object count
        cv::putText(frame, "Objects: " + std::to_string(indices.size()),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.7f, cv::Scalar(0, 255, 0), 2);

        cv::imshow("YOLOv8 Detection", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}