import cv2
import pyzed.sl as sl

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

class CameraNode():
    def __init__(self):
        self.InitCamera()
    
    def InitCamera(self):
        init = sl.InitParameters()
        self.cam = sl.Camera()
        if not self.cam.is_opened():
            print("Opening ZED Camera...")
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        
        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()

        self.print_camera_information()

    def print_camera_information(self):
        print("Resolution: {0}, {1}.".format(round(self.cam.get_camera_information().camera_resolution.width, 2),
                                             self.cam.get_camera_information().camera_resolution.height))
        print("Camera FPS: {0}.".format(self.cam.get_camera_information().camera_fps))
        print("Firmware: {0}.".format(self.cam.get_camera_information().camera_firmware_version))
        print("Serial number: {0}.\n".format(self.cam.get_camera_information().serial_number))
    
    def TakeImage(self):
        img_data = None
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_image(self.mat, sl.VIEW.LEFT)
            img_data = self.mat.get_data()
        return img_data