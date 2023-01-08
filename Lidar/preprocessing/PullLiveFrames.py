
#import innopy models
import numpy as np
from innopy.api import DeviceInterface, FileReader, FrameDataAttributes, GrabType
from sklearn.preprocessing import minmax_scale
import open3d as o3d
import argparse
import os
import time
from lidar_record_preprocess import PointCloudCropper
#import patchwork modules
import sys
import pypatchworkpp

target_path = "../../../patchwork-plusplus/build/python_wrapper"

try:
    patchwork_module_path = target_path  
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

#define the handler class and callback function
class PointCloudPatchworkHandler:
    def __init__(self):
        self.attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0), FrameDataAttributes(GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA)]
        config_files_path = '../lidar_configuration_files'
        #This config file will remove the blooming pixles.
        self.di = DeviceInterface(config_file_name=config_files_path+'/om_remove_blooming_config.json', is_connect=False)
        # Patchwork++ initialization
        params = pypatchworkpp.Parameters()         #change sensor heights,elevation
        params.verbose = True
        params.sensor_height = 0.8
        params.elevation_thr= [-0.8,-0.2,0.2,0.8]
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        for i in range(len(self.attr)):
            self.di.activate_buffer(self.attr[i], True)

    def callback(self, h):
        try:
            frame = self.di.get_frame(self.attr)
            if frame.success:
                mes = frame.results['GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0']
                meta = frame.results['GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA']
                frame_num = frame.frame_number
                vertices = []
                for pixel_num in range(len(mes)):
                    if mes['confidence'][pixel_num] > 0 and meta['ghost'][pixel_num] == 0 and meta['noise'][pixel_num] == 0:
                        vertices.append([mes['x'][pixel_num], mes['y'][pixel_num], mes['z'][pixel_num], mes['reflectivity'][pixel_num]])
                vertices_np = np.asarray(vertices)
                vertices_np /= 100 # convert to meters
                croped_vertices = PointCloudCropper(vertices_np)
                    # Estimate Ground
                self.PatchworkPLUSPLUS.estimateGround(croped_vertices)

                 # Get Ground and Nonground
                ground      = self.PatchworkPLUSPLUS.getGround()
                nonground   = self.PatchworkPLUSPLUS.getNonground()
                time_taken  = self.PatchworkPLUSPLUS.getTimeTaken()

                # Get centers and normals for patches
                centers     = self.PatchworkPLUSPLUS.getCenters()
                normals     = self.PatchworkPLUSPLUS.getNormals()
                print("Origianl Points  #: ", croped_vertices.shape[0])
                print("Ground Points    #: ", ground.shape[0])
                print("Nonground Points #: ", nonground.shape[0])
                print("Time Taken : ", time_taken / 1000000, "(sec)")
                print("Press ... \n")
                print("\t H  : help")
                print("\t N  : visualize the surface normals")
                print("\tESC : close the Open3D window")

        except:
            print('PointCloudPatchworkHandler: failed executing frame')

    def register_patchwork_handler(self):
        # Register callback
        self.di.register_new_frame_callback(self.callback) 

    def finish(self):
        self.di.device_close()


def main():

    fh = PointCloudPatchworkHandler()
    fh.register_patchwork_handler()
    time.sleep(20)
    print("The End:)")
    fh.finish()

if __name__ == '__main__':
    main()



