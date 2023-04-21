#include "ImagePreprocessing/ImagePreprocessing.h"

class ImagePreprocessingNode : public rclcpp::Node 
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr processed_image_pub_;
    std::shared_ptr<ImagePreprocessing> image_preprocessing_;

    void processImage(const sensor_msgs::msg::Image::SharedPtr& msg);

public:
    ImagePreprocessingNode();
};

