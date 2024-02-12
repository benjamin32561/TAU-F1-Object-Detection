import numpy as np
from innopy.api import FileReader, FrameDataAttributes, GrabType
from sklearn.preprocessing import minmax_scale
import open3d as o3d
import argparse
import os
import queue
import sys
#build path
target_path = "/home/idola/PycharmProjects/patchwork-plusplus/build/python_wrapper"

try:
    patchwork_module_path = target_path
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)


class FrameParameters:
    def __init__(self, ground, nonground, timetaken, centers, normals):
        self.ground = ground
        self.nonground = nonground
        self.timetaken = timetaken
        self.centers = centers
        self.normals = normals


class PointCloudCropper(object):
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def __call__(self, point_cloud):
        return point_cloud[(point_cloud[:, 0] > self.x_min) & (point_cloud[:, 0] < self.x_max) & (point_cloud[:, 1] > self.y_min) & (point_cloud[:, 1] < self.y_max) & (point_cloud[:, 2] > self.z_min) & (point_cloud[:, 2] < self.z_max)]


class Patchwork_init:
    def __init__(self):
        #Patchwork++ initialization
        params = pypatchworkpp.Parameters()         #change sensor heights,elevation
        params.verbose = True
        params.sensor_height = 0.8
        params.elevation_thr= [-0.8,-0.2,0.2,0.8]
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
        self.pcc = PointCloudCropper(x_min=3, x_max=15, y_max=10, y_min=-10, z_max=3, z_min=-1)