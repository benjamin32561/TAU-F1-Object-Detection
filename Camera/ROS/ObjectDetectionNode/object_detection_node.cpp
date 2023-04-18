#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Empty.h>

void imageCallback(const sensor_msgs::Image& image)
{
    // TODO: Implement object detection on the image
    // ...

    // Publish empty message to detection topic
    static ros::NodeHandle nh;
    static ros::Publisher detection_pub = nh.advertise<std_msgs::Empty>("detection", 1);
    std_msgs::Empty empty_msg;
    detection_pub.publish(empty_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detection_node");

    // Subscribe to camera topic
    static ros::NodeHandle nh;
    static ros::Subscriber image_sub = nh.subscribe("image_topic", 1, imageCallback);

    ros::spin();
    return 0;
}
