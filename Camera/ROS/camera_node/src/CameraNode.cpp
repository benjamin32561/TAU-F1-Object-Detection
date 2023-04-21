#include "CameraNode.h"

CameraNode::CameraNode() : Node("camera_node") 
{
    // Create the publisher for the image topic
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("image_topic", 10);

    // Initialize the ZED camera
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION_HD1080;
    init_params.camera_fps = 30;
    zed_.open(init_params);
}

void CameraNode::captureImage() 
{
    // Capture a new frame from the ZED camera
    sl::Mat image_zed;
    zed_.grab();
    zed_.retrieveImage(image_zed, sl::VIEW_LEFT);

    // Convert the image format from ZED to OpenCV
    cv::Mat image_cv(image_zed.getHeight(), image_zed.getWidth(),
                        CV_8UC4, image_zed.getPtr<sl::uchar1>(sl::MEM_CPU));

    // Create a sensor_msgs/Image message from the OpenCV image
    sensor_msgs::msg::Image::SharedPtr image_msg =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_cv).toImageMsg();

    // Publish the image message to the image topic
    image_pub_->publish(*image_msg);
}