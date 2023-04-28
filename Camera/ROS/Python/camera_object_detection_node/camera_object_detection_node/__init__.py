#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2

class ObjectDetectionModel():
    def __init__(self):
        self.model = None
    
    def DetectObjects(self, img):
        detections = None
        return detections

class ImagePreprocessingNode(Node):
    def __init__(self):
        super().__init__('camera_object_detection_node')
        self.subscriber_ = self.create_subscription(
            Image,
            'processed_image',
            self.DetectObjectcs,
            10
        )
        self.publisher_ = self.create_publisher(String, 'camera_detections', 10)
        self.model = ObjectDetectionModel()

    def DetectObjectcs(self, msg):
        final_msg = String()
        processed_img = msg
        detections = self.model.DetectObjects(processed_img)
        final_msg.data = ""
        self.publisher_.publish(final_msg)

def main(args=None):
    print('Started camera_object_detection_node')
    rclpy.init(args=args)
    image_processing_node = ImagePreprocessingNode()
    rclpy.spin(image_processing_node)
    image_processing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
