
#import innopy models
import numpy as np
from innopy.api import DeviceInterface, FileReader, FrameDataAttributes, GrabType
from sklearn.preprocessing import minmax_scale
import open3d as o3d
import argparse
import os
import time
from ..common.classes import FrameParameters, PointCloudCropper, Patchwork_init
#import patchwork modules
import sys
#import pypatchworkpp
import queue
import threading


#build path
target_path = "/home/idola/PycharmProjects/patchwork-plusplus/build/python_wrapper"

try:
    patchwork_module_path = target_path
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

# Create a queue
frame_queue = queue.Queue()

#define the handler class and callback function
class PointCloudPatchworkHandler:
    def __init__(self):
        self.attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0), FrameDataAttributes(GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA)]
        config_files_path = '/home/idola/PycharmProjects/TAU-F1-Object-Detection/Lidar/innoviz_api/examples/lidar_configuration_files'
        #This config file will remove the blooming pixles.
        self.di = DeviceInterface(config_file_name=config_files_path+'/om_remove_blooming_config.json', is_connect=False)
         #Patchwork++ initialization
        params = pypatchworkpp.Parameters()         #change sensor heights,elevation
        params.verbose = True
        params.sensor_height = 0.8
        params.elevation_thr= [-0.8,-0.2,0.2,0.8]
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

        for i in range(len(self.attr)):
            self.di.activate_buffer(self.attr[i], True)

        self.pcc = PointCloudCropper(x_min=3, x_max=15, y_max=10, y_min=-10, z_max=3, z_min=-1)

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
                croped_vertices = self.pcc(vertices_np)
                    # Estimate Ground
                self.PatchworkPLUSPLUS.estimateGround(croped_vertices)
                print("estimate ground")
                 # Get Ground and Nonground
                ground      = self.PatchworkPLUSPLUS.getGround()
                nonground   = self.PatchworkPLUSPLUS.getNonground()
                time_taken  = self.PatchworkPLUSPLUS.getTimeTaken()

                # Get centers and normals for patches
                centers     = self.PatchworkPLUSPLUS.getCenters()
                normals     = self.PatchworkPLUSPLUS.getNormals()
                #print("Origianl Points  #: ", croped_vertices.shape[0])
                #print("Ground Points    #: ", ground.shape[0])
                #print("Nonground Points #: ", nonground.shape[0])
                #print("Time Taken : ", time_taken / 1000000, "(sec)")
                #print("Press ... \n")
                #print("\t H  : help")
                #print("\t N  : visualize the surface normals")
                #print("\tESC : close the Open3D window")
                # Add the frame parameters to the queue
                frame_params = FrameParameters(ground, nonground, time_taken, centers, normals)
                frame_queue.put(frame_params)

        except Exception as err:
            print(err)
            print('PointCloudPatchworkHandler: failed executing frame')

    def register_patchwork_handler(self):
        # Register callback
        self.di.register_new_frame_callback(self.callback)

    def init_frame(self, vis, frame_params):
        # Visualize
        #vis = o3d.visualization.VisualizerWithKeyCallback()
        #print("vis1")
        #vis.create_window(width=600, height=400)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        ground_o3d = o3d.geometry.PointCloud()
        ground_o3d.points = o3d.utility.Vector3dVector(frame_params.ground)
        print(frame_params.ground.shape[0])
        print(frame_params.nonground.shape[0])
        print(frame_params.centers.shape[0])

        if frame_params.ground.shape[0] != 0:
            ground_o3d.colors = o3d.utility.Vector3dVector(
                np.array([[0.0, 1.0, 0.0] for _ in range(frame_params.ground.shape[0])], dtype=float))  # RGB
        print("vis2\n")

        nonground_o3d = o3d.geometry.PointCloud()
        nonground_o3d.points = o3d.utility.Vector3dVector(frame_params.nonground)
        if frame_params.nonground.shape[0] != 0:
            nonground_o3d.colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 0.0, 0.0] for _ in range(frame_params.nonground.shape[0])], dtype=float))  # RGB
        print("vis3\n")

        centers_o3d = o3d.geometry.PointCloud()
        centers_o3d.points = o3d.utility.Vector3dVector(frame_params.centers)
        centers_o3d.normals = o3d.utility.Vector3dVector(frame_params.normals)
        if frame_params.centers.shape[0] != 0 :
            centers_o3d.colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 1.0, 0.0] for _ in range(frame_params.centers.shape[0])], dtype=float))  # RGB

        vis.add_geometry(mesh)
        vis.add_geometry(ground_o3d)
        vis.add_geometry(nonground_o3d)
        vis.add_geometry(centers_o3d)
        print("vis4\n")
        #vis.poll_events()
        #vis.update_renderer()
        #vis.run()
        print("vis4\n")

    def update_frame(self,vis, frame_params):
        #ground_o3d = o3d.geometry.PointCloud()
        #ground_o3d.points = o3d.utility.Vector3dVector(frame_params.ground)
        #nonground_o3d = o3d.geometry.PointCloud()
        #nonground_o3d.points = o3d.utility.Vector3dVector(frame_params.nonground)
        #centers_o3d = o3d.geometry.PointCloud()
        #centers_o3d.points = o3d.utility.Vector3dVector(frame_params.centers)
        #centers_o3d.normals = o3d.utility.Vector3dVector(frame_params.normals)
        #vis.update_geometry(ground_o3d)
        #vis.update_geometry(nonground_o3d)
        #vis.update_geometry(centers_o3d)
        vis.poll_events()
        vis.update_renderer()
    def finish(self):
        self.di.device_close()


def main():
    i = 0
    fh = PointCloudPatchworkHandler()
    fh.register_patchwork_handler()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=600, height=400)
    #time.sleep(20)
    frame_params = frame_queue.get(timeout=1)  # Timeout is optional
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(frame_params.ground)

    if frame_params.ground.shape[0] != 0:
        ground_o3d.colors = o3d.utility.Vector3dVector(
            np.array([[0.0, 1.0, 0.0] for _ in range(frame_params.ground.shape[0])], dtype=float))  # RGB

    nonground_o3d = o3d.geometry.PointCloud()
    nonground_o3d.points = o3d.utility.Vector3dVector(frame_params.nonground)
    if frame_params.nonground.shape[0] != 0:
        nonground_o3d.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 0.0, 0.0] for _ in range(frame_params.nonground.shape[0])], dtype=float))  # RGB

    centers_o3d = o3d.geometry.PointCloud()
    centers_o3d.points = o3d.utility.Vector3dVector(frame_params.centers)
    centers_o3d.normals = o3d.utility.Vector3dVector(frame_params.normals)
    if frame_params.centers.shape[0] != 0:
        centers_o3d.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 1.0, 0.0] for _ in range(frame_params.centers.shape[0])], dtype=float))  # RGB

    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)
    vis.add_geometry(centers_o3d)
    while i < 100000:
        try:
            frame_params = frame_queue.get(timeout=1)  # Timeout is optional
            ground_o3d.points = o3d.utility.Vector3dVector(frame_params.ground)
            if frame_params.ground.shape[0] != 0:
                ground_o3d.colors = o3d.utility.Vector3dVector(
                    np.array([[0.0, 1.0, 0.0] for _ in range(frame_params.ground.shape[0])], dtype=float))  # RGB
            nonground_o3d.points = o3d.utility.Vector3dVector(frame_params.nonground)
            if frame_params.nonground.shape[0] != 0:
                nonground_o3d.colors = o3d.utility.Vector3dVector(
                    np.array([[1.0, 0.0, 0.0] for _ in range(frame_params.nonground.shape[0])], dtype=float))  # RGB
            centers_o3d.points = o3d.utility.Vector3dVector(frame_params.centers)
            centers_o3d.normals = o3d.utility.Vector3dVector(frame_params.normals)
            if frame_params.centers.shape[0] != 0:
                centers_o3d.colors = o3d.utility.Vector3dVector(
                    np.array([[1.0, 1.0, 0.0] for _ in range(frame_params.centers.shape[0])], dtype=float))  # RGB
            vis.update_geometry(ground_o3d)
            vis.update_geometry(nonground_o3d)
            vis.update_geometry(centers_o3d)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
            i += 1
        except queue.Empty:
            print("queue is empty, breaking loop")
            break
    vis.destroy_window()
    print("The End:)")
    fh.finish()


# Create a thread for the main function
main_thread = threading.Thread(target=main)

# Start the main thread
main_thread.start()

if __name__ == '__main__':
    main()



