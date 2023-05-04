#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>

class Model 
{
public:
    Model(const std::string& filename);
    ~Model();

    void Detect();
};
