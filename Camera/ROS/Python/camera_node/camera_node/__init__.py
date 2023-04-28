#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import pyzed.sl as sl

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(String, 'camera_topic', 10)

    def run(self):
        while True:
            msg = String()
            msg.data = 'Hello, world!'
            self.publisher_.publish(msg)
            self.get_logger().info('Published message')

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    camera_node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
