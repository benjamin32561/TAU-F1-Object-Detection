#include "CameraNode.h"

CameraNode::CameraNode(ros::NodeHandle& nh)
    : nh_(nh)
{
    // Initialize ZED camera
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION_HD720;
    init_params.camera_fps = 30;
    zed_.open(init_params);

    // Initialize ROS publisher
    image_pub_ = nh_.advertise<sensor_msgs::Image>("image_topic", 1);

    // Initialize ROS timer
    timer_ = nh_.createTimer(ros::Duration(1.0 / 30), &CameraNode::imageCallback, this);
}

CameraNode::~CameraNode()
{
    // Cleanup
    zed_.close();
}

void CameraNode::imageCallback(const ros::TimerEvent& event)
{
    // Capture image from ZED camera
    sl::Mat zed_image;
    zed_.grab();
    zed_.retrieveImage(zed_image, sl::VIEW_LEFT);

    // Create ROS message from ZED image
    sensor_msgs::Image ros_image;
    ros_image.header.stamp = ros::Time::now();
    ros_image.header.frame_id = "camera_frame";
    ros_image.height = zed_image.getHeight();
    ros_image.width = zed_image.getWidth();
    ros_image.encoding = "rgb8";
    ros_image.step = zed_image.getStep();
    ros_image.data.assign(zed_image.getPtr<sl::uchar1>(), zed_image.getPtr<sl::uchar1>() + ros_image.height * ros_image.step);

    // Publish ROS message
    image_pub_.publish(ros_image);
}
