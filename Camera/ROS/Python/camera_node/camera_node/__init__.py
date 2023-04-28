#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

import cv2
import pyzed.sl as sl

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

class Camera():
    def __init__(self):
        self.InitCamera()
        self.print_camera_information()
    
    def InitCamera(self):
        init = sl.InitParameters()
        self.cam = sl.Camera()
        if not self.cam.is_opened():
            print("Opening ZED Camera...")
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            raise Exception("Can Not Open ZED Camera")
        
        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()

    def print_camera_information(self):
        print("Resolution: {0}, {1}.".format(round(self.cam.get_camera_information().camera_resolution.width, 2),
                                             self.cam.get_camera_information().camera_resolution.height))
        print("Camera FPS: {0}.".format(self.cam.get_camera_information().camera_fps))
        print("Firmware: {0}.".format(self.cam.get_camera_information().camera_firmware_version))
        print("Serial number: {0}.\n".format(self.cam.get_camera_information().serial_number))
    
    def GetImage(self):
        img_data = None
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_image(self.mat, sl.VIEW.LEFT)
            img_data = self.mat.get_data()
        return img_data

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
