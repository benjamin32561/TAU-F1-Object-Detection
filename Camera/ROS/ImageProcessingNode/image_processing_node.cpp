#include "ImagePreprocessingNode.h"

// compile:
// g++ image_processing_node.cpp -o ~/TAU-F1-Object-Detection/Camera/ROS/image_processing_node.out

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ImagePreprocessingNode>();

    while (rclcpp::ok()) {
        node->captureImage();
        rclcpp::spin_some(node);
    }

    rclcpp::shutdown();
    return 0;
}