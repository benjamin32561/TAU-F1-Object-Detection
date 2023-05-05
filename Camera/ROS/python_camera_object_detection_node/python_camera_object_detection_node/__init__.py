#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cv2 import resize, imread

from python_camera_object_detection_node.Camera import Camera
from python_camera_object_detection_node.Models import YOLOv5

from python_camera_object_detection_node.Helper import ListFilesInFolder, GetRandomFromList

from time import sleep

PUBLISHER_NAME = 'camera_object_detection'

YOLOv5 = "YOLOv5"
YOLOv5_MODEL_PATH = "/mnt/c/Users/ben32/Desktop/Models/YOLOv5.pt"

TESTING = True
TEST_IMGS_FOLDER_PATH = "/mnt/c/Users/ben32/Desktop/tau f images/"
TEST_IMGS_PATHS = ListFilesInFolder(TEST_IMGS_FOLDER_PATH)

class CameraObjectDetectionNode(Node):
    def __init__(self,model=YOLOv5):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(String, PUBLISHER_NAME, 10)
        try:
            self.camera = Camera()
        except Exception as e:
            self.get_logger().error(f"Failed to load Camera: {e}")
            self.camera = None
        self.model = None

        try:
            if model==YOLOv5:
                self.model = YOLOv5(model_path=YOLOv5_MODEL_PATH)
        except Exception as e:
            self.get_logger().error(f"Failed to load model {model}: {e}")
        

    def run(self):
        img = None
        if self.camera is not None:
            img = self.camera.GetImage()
        if img is not None:
            # preprocess image
            img = resize(img,self.model.image_dimensions)

            # detect objects
            detections = self.model.DetectObjects(img)

            # postprocess detections
            msg = String()
            msg.data = ''
            self.publisher_.publish(msg)

        else:
            if not TESTING:
                self.get_logger().info('img is None')
            else:
                self.get_logger().info('In Test Mode, Failed To Get Image From Camera')
                img = imread(GetRandomFromList(TEST_IMGS_PATHS))[..., ::-1] # reading and converting from BGR to RGB
                img = resize(img,self.model.image_dimensions) # resizing image to match model
                results = self.model.DetectObjects(img,True)
                sleep(60)

def main(args=None):
    print('Started')
    rclpy.init(args=args)
    camera_obj_ddt_node = CameraObjectDetectionNode(YOLOv5)
    rclpy.spin(camera_obj_ddt_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
