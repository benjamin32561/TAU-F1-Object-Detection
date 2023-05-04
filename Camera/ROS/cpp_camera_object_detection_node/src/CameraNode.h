#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

class CameraNode : public rclcpp::Node 
{
private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    std::shared_ptr<sensor_msgs::msg::Image> image_msg_;
    sl::Camera zed_;
    rclcpp::TimerBase::SharedPtr timer_ = create_wall_timer(std::chrono::milliseconds(1000), std::bind(&CameraNode::captureAndPublishImage, this));

    void captureAndPublishImage();

public:
    CameraNode();
    ~CameraNode();

};