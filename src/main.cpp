#include "model.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(){
    ObjectDetectionModel model ("../models/yolov5s.onnx", "../coco.names");
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) {
    std::cout << "cannot open camera";
    
    }
    std::chrono::time_point<std::chrono::system_clock> lastUpdate;
    while (true) {
        model << cap;
    }
}