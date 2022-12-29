import cv2
import pyzed.sl as sl
import math
import numpy as np
import sys

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 50
    runtime_parameters.textureness_confidence_threshold = 100

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    key = ''
    while key != 113:  # for 'q' key
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv2.imshow("ZED", image.get_data())
            key = cv2.waitKey(5)
            
            zed.retrieve_measure(point_cloud, sl.MEASURE.DEPTH)
            cv2.imshow("ZED_DEPTH", point_cloud.get_data())
        else:
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    zed.close()

if __name__ == "__main__":
    main()
