#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from python_camera_object_detection_node.Camera import Camera

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)
        try:
            self.camera = Camera()
        except Exception as e:
            print(e)
            self.camera = None
        self.bridge = CvBridge()

    def run(self):
        while True:
            img = None
            if self.camera is not None:
                img = self.camera.GetImage()
            if img is not None:
                msg = self.bridge.cv2_to_imgmsg(img)
                self.publisher_.publish(msg)
            else:
                self.get_logger().info('img is None')

def main(args=None):
    print('Started')
    rclpy.init(args=args)
    camera_node = CameraNode()
    camera_node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
