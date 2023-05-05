#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cv2 import resize

from python_camera_object_detection_node.Camera import Camera
from python_camera_object_detection_node.Models import YOLOv5

PUBLISHER_NAME = 'camera_object_detection'
YOLOv5 = "YOLOv5"

class CameraObjectDetectionNode(Node):
    def __init__(self,model=YOLOv5):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(String, PUBLISHER_NAME, 10)
        try:
            self.camera = Camera()
        except Exception as e:
            print(e)
            self.camera = None
        self.model = None

        if model==YOLOv5:
            self.model = YOLOv5('')

    def run(self):
        img = None
        if self.camera is not None:
            img = self.camera.GetImage()
        if img is not None:
            # preprocess image
            img = resize(img,self.model.image_dimensions)

            # detect objects
            detections = self.model.DetectObbjects(img)

            # postprocess detections
            msg = String()
            msg.data = ''
            self.publisher_.publish(msg)

        else:
            self.get_logger().info('img is None')

def main(args=None):
    print('Started')
    rclpy.init(args=args)
    camera_obj_ddt_node = CameraObjectDetectionNode(YOLOv5)
    rclpy.spin(camera_obj_ddt_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
