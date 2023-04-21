#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

class CameraNode : public rclcpp::Node 
{
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    sl::Camera zed_;

    void captureImage();

public:
    CameraNode();
};