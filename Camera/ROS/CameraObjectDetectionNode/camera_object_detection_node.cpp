#include "model/model.h"

// compile:
// g++ object_detection_node.cpp -o ~/TAU-F1-Object-Detection/Camera/ROS/object_detection_node.out

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraObjectDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}