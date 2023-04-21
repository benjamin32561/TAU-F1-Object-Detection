#include "Model/model.h"

class CameraObjectDetectionNode : public rclcpp::Node 
{
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr processed_image_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr camera_model_output_pub_;
    std::shared_ptr<Model> model_;

    void detectObjects(const sensor_msgs::msg::Image::SharedPtr& msg);
    
public:
    CameraObjectDetectionNode();

};