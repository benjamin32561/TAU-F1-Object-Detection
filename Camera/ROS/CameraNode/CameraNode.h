#ifndef CAMERA_NODE_H
#define CAMERA_NODE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sl/Camera.hpp>

class CameraNode
{
public:
    CameraNode(ros::NodeHandle& nh);
    ~CameraNode();

private:
    void imageCallback(const ros::TimerEvent& event);

    ros::NodeHandle nh_;
    ros::Publisher image_pub_;
    ros::Timer timer_;
    sl::Camera zed_;
};

#endif