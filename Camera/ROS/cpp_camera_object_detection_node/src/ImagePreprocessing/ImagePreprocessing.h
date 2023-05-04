#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class ImagePreprocessing 
{
public:
    ImagePreprocessing();

    sensor_msgs::msg::Image::SharedPtr preprocess(const sensor_msgs::msg::Image::SharedPtr& msg);
};