#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include <limits>
// For disable PCL complile lib, to use PointXYZIR
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <signal.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/MarkerArray.h>

#include "patchworkpp/patchworkpp.hpp"

using PointType = PointXYZIRingSectorZone;
using namespace std;

boost::shared_ptr<PatchWorkpp<PointType>> PatchworkppGroundSeg;

ros::Publisher pub_cloud;
ros::Publisher pub_ground;
ros::Publisher pub_non_ground;
ros::Publisher pub_markers;

template<typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, const ros::Time& stamp, std::string frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.stamp = stamp;
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}

visualization_msgs::MarkerArray publishBoundingBoxes(const pcl::PointCloud<PointType>& cloud, const ros::Time& stamp, const std::string& frame_id) {
    std::map<std::tuple<int, int, int>, pcl::PointCloud<PointType>> groups;
    for (const auto& point : cloud) {
        groups[std::make_tuple(point.zone, point.sector, point.ring)].push_back(point);
    }

    visualization_msgs::MarkerArray marker_array;
    int id = 0;
    for (const auto& group : groups) {
        auto& points = group.second;
        if (points.size() < 50 || points.size() > 200) {
            continue;  // Skip processing this group
        }
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(points, min_pt, max_pt);

        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = stamp;
        marker.ns = "bounding_boxes";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = (min_pt.x() + max_pt.x()) / 2;
        marker.pose.position.y = (min_pt.y() + max_pt.y()) / 2;
        marker.pose.position.z = min_pt.z();
        marker.scale.x = (max_pt.x() - min_pt.x());
        marker.scale.y = (max_pt.y() - min_pt.y());
        marker.scale.z = (max_pt.z() - min_pt.z());
        marker.color.r = 0.0f;
        marker.color.g = 0.0f;
        marker.color.b = 1.0f;
        marker.color.a = 0.5;
        marker.lifetime = ros::Duration(0.1);
        marker_array.markers.push_back(marker);
         // Calculate the distance to the center of the bounding box
        double center_x = marker.pose.position.x;
        double center_y = marker.pose.position.y;
        double center_z = marker.pose.position.z;
        double distance = std::sqrt(center_x * center_x + center_y * center_y + center_z * center_z);
        // Log the information
        ROS_INFO_STREAM("Bounding Box " << id - 1 << ": Center ("
                        << center_x << ", " << center_y << ", " << center_z << "), Distance: "
                        << distance << " meters");
    }
    return marker_array;
    // print marker_array

}

void callbackCloud(const sensor_msgs::PointCloud2::Ptr &cloud_msg)
{

    ros::Time time_now = ros::Time::now();
    // log it

    double time_taken;
    pcl::PointCloud<pcl::PointXYZI> cloud_xyz;
    pcl::PointCloud<PointType> pc_curr;
    pcl::PointCloud<PointType> pc_ground;
    pcl::PointCloud<PointType> pc_non_ground;

    pcl::fromROSMsg(*cloud_msg, cloud_xyz);
    pc_curr.header = cloud_xyz.header;  // Copy header
    pc_curr.points.resize(cloud_xyz.points.size());
    for (size_t i = 0; i < cloud_xyz.points.size(); ++i) {
        auto& src_pt = cloud_xyz.points[i];
        auto& dst_pt = pc_curr.points[i];
        dst_pt.x = src_pt.x;
        dst_pt.y = src_pt.y;
        dst_pt.z = src_pt.z;
        dst_pt.intensity = src_pt.intensity;
    }

    PatchworkppGroundSeg->estimate_ground(pc_curr, pc_ground, pc_non_ground, time_taken);

    ROS_INFO_STREAM("\033[1;32m" << "Input PointCloud: " << pc_curr.size() << " -> Ground: " << pc_ground.size() <<  "/ NonGround: " << pc_non_ground.size()
         << " (running_time: " << time_taken << " sec)" << "\033[0m");
    auto marker_array = publishBoundingBoxes(pc_non_ground, cloud_msg->header.stamp, cloud_msg->header.frame_id);

    pub_cloud.publish(cloud2msg(pc_curr, cloud_msg->header.stamp, cloud_msg->header.frame_id));
    pub_ground.publish(cloud2msg(pc_ground, cloud_msg->header.stamp, cloud_msg->header.frame_id));
    pub_non_ground.publish(cloud2msg(pc_non_ground, cloud_msg->header.stamp, cloud_msg->header.frame_id));
    pub_markers.publish(marker_array);

    // get the ros time now
    // log it
    ROS_INFO_STREAM("Time now: " << time_now - ros::Time::now());
}

int main(int argc, char**argv) {

    ros::init(argc, argv, "Demo");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string cloud_topic;
    pnh.param<string>("cloud_topic", cloud_topic, "/pointcloud");

    cout << "Operating patchwork++..." << endl;
    PatchworkppGroundSeg.reset(new PatchWorkpp<PointType>(&pnh));

    pub_cloud       = pnh.advertise<sensor_msgs::PointCloud2>("cloud", 100, true);
    pub_ground      = pnh.advertise<sensor_msgs::PointCloud2>("ground", 100, true);
    pub_non_ground  = pnh.advertise<sensor_msgs::PointCloud2>("nonground", 100, true);
    pub_markers = pnh.advertise<visualization_msgs::MarkerArray>("bounding_boxes", 100, true);

    ros::Subscriber sub_cloud = nh.subscribe(cloud_topic, 100, callbackCloud);
    
    ros::spin();

    return 0;
}
