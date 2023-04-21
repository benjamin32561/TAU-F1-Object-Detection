#include <iostream>
#include "CameraNode.h"

// compile:
// g++ camera_node.cpp -o ~/TAU-F1-Object-Detection/Camera/ROS/camera_node.out

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();

    rclcpp::Rate loop_rate(30);
    while (rclcpp::ok()) {
        node->captureImage();
        rclcpp::spin_some(node);
        loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}