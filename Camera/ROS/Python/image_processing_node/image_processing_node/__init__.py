#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

FINAL_IMG_W = 640
FINAL_IMG_H = 640

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
        self.bridge = CvBridge()

    def ProcessImage(self, msg):
        # convert the message to an OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # resize the image to 640x640
        img_resized = cv2.resize(img, (FINAL_IMG_W, FINAL_IMG_H))
        # convert the resized image back to an Image message
        img_msg = self.bridge.cv2_to_imgmsg(img_resized, encoding='bgr8')
        # publish the resized image message to the 'processed_image' topic
        self.publisher.publish(img_msg)

def main(args=None):
    print('Started')
    rclpy.init(args=args)
    image_processing_node = ImagePreprocessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
