#include <stdin>

int main()
{
    std::cout << "nice!" << std::endl;
    
    return 0;
}

// #include <ros/ros.h>
// #include <sensor_msgs/Image.h>
// #include "CameraNode.h"

// int main(int argc, char** argv)
// {
//     // Initialize ROS node
//     ros::init(argc, argv, "camera_node");
//     ros::NodeHandle nh;

//     // Create camera node
//     CameraNode camera_node(nh);

//     // Spin ROS node
//     ros::spin();

//     return 0;
// }
