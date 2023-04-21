#include "CameraNode.h"

// compile:
// g++ camera_node.cpp -o ~/TAU-F1-Object-Detection/Camera/ROS/camera_node.out

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}