#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class ImagePreprocessing 
{
public:
    ImagePreprocessing();

    void PreprocessImage();
};