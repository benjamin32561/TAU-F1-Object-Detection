#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

ObjectDetectionNode::ObjectDetectionNode() : Node("camera_node") 
{
    // Create the publisher for the image topic
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("image_topic", 10);

    // Initialize the ZED camera
    zed_.open();
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = 30;
    init_params.coordinate_units = sl::UNIT::METER;
    zed_.open(init_params);

    // Set up the image message
    image_msg_ = std::make_shared<sensor_msgs::msg::Image>();
    image_msg_->header.frame_id = "camera_frame";
    image_msg_->height = zed_.getCameraInformation().camera_resolution.height;
    image_msg_->width = zed_.getCameraInformation().camera_resolution.width;
    image_msg_->encoding = "bgr8";
    image_msg_->is_bigendian = false;
    image_msg_->step = zed_.getCameraInformation().camera_resolution.width * 3;
}

ObjectDetectionNode::~ObjectDetectionNode()
{
    // Close the ZED camera
    zed_.close();
}

void ObjectDetectionNode::captureAndPublishImage() 
{
    // Capture a new image from the ZED camera
    sl::Mat image;
    zed_.grab();
    zed_.retrieveImage(image, sl::VIEW::LEFT);

    // Copy the image data to the image message
    memcpy(&image_msg_->data[0], image.getPtr<sl::uchar1>(sl::MEM::CPU), image_msg_->step * image_msg_->height);

    // Publish the image message
    image_msg_->header.stamp = this->now();
    image_pub_->publish(*image_msg_);
}