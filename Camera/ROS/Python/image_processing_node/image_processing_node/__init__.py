#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2

class ImagePreprocessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        self.subscriber_ = self.create_subscription(
            Image,
            'image_topic',
            self.ProcessImage,
            10
        )
        self.publisher_ = self.create_publisher(Image, 'processed_image', 10)

    def ProcessImage(self, msg):
        final_msg = Image()
        self.publisher_.publish(final_msg)

def main(args=None):
    print('Started')
    rclpy.init(args=args)
    image_processing_node = ImagePreprocessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
