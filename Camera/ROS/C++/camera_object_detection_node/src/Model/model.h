#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

class Model 
{
private:
    std::shared_ptr<torch::jit::script::Module> model_;

public:
    Model(const std::string& filename);
    ~Model();

    void Detect();
};
