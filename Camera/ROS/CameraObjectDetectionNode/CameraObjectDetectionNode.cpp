#include "CameraObjectDetectionNode.h"

CameraObjectDetectionNode::CameraObjectDetectionNode() : Node("camera_object_detection_node") 
{
    // Create the subscriber for the processed image topic
    processed_image_sub_ = create_subscription<sensor_msgs::msg::Image>("processed_image", 10,
                                                                            std::bind(&CameraObjectDetectionNode::detectObjects, this, std::placeholders::_1));

    // Create the publisher for the object detection results topic
    camera_model_output_pub_ = create_publisher<std_msgs::msg::String>("camera_model_output", 10);

    // Initialize the model
    model_ = std::make_shared<Model>();
}

void CameraObjectDetectionNode::detectObjects(const sensor_msgs::msg::Image::SharedPtr& msg) 
{
    // Convert the image message to an OpenCV Mat
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("CameraObjectDetectionNode"), "cv_bridge exception: " << e.what());
        return;
    }

    // Run object detection on the image
    std::vector<std::string> objects = model_->detect(cv_ptr->image);

    // Publish the object detection results as a comma-separated string
    std_msgs::msg::String::SharedPtr output_msg = std_msgs::msg::String::SharedPtr(new std_msgs::msg::String());
    output_msg->data = boost::algorithm::join(objects, ",");
    camera_model_output_pub_->publish(*output_msg);
}