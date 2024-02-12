import numpy as np
from innopy.api import FileReader, FrameDataAttributes, GrabType
from sklearn.preprocessing import minmax_scale
import open3d as o3d
import argparse
import os
import queue
from ..common.classes import FrameParameters, PointCloudCropper, Patchwork_init

#build path
target_path = "/home/alergand/code/patchwork-plusplus/build/python_wrapper"

try:
    patchwork_module_path = target_path
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)


def split_lidar_record_to_frames(record_path,num_of_frames, format, output_path,pcd_cropper, frame_queue, normalize=False):
    attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0),FrameDataAttributes(GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA)]
    config_path = '/home/alergand/code/TAU-F1-Object-Detection/Lidar/preprocessing/InnovizAPI/examples/lidar_configuration_files/recording_remove_blooming_config.json'
    frames = FileReader('/Lidar/processing/recordings', num_of_cores=1, config_filepath=config_path)
    number_of_frames = min(num_of_frames, frames.num_of_frames) if num_of_frames != -1 else frames.num_of_frames
    for i in range(number_of_frames):
        frame = frames.get_frame(i, attr)
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
            croped_vertices = Patchwork_init.pcc(vertices_np)
            print("estimate ground")
            # Get Ground and Nonground
            ground      = Patchwork_init.PatchworkPLUSPLUS.getGround()
            nonground   = Patchwork_init.PatchworkPLUSPLUS.getNonground()
            time_taken  = Patchwork_init.PatchworkPLUSPLUS.getTimeTaken()

            # Get centers and normals for patches
            centers     = Patchwork_init.PatchworkPLUSPLUS.getCenters()
            normals     = Patchwork_init.PatchworkPLUSPLUS.getNormals()
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
            if normalize:
                norm_vertices_x = minmax_scale(croped_vertices[:, 0].reshape(-1,1), feature_range=(0, 1))
                norm_vertices_y = minmax_scale(croped_vertices[:, 1].reshape(-1,1), feature_range=(-1, 1))
                norm_vertices_z = minmax_scale(croped_vertices[:, 2].reshape(-1, 1), feature_range=(0, 1))
                croped_vertices = np.concatenate((norm_vertices_y,norm_vertices_z, norm_vertices_x), axis=1)
            else:
                croped_vertices = np.concatenate((croped_vertices[:, 1].reshape(-1,1),croped_vertices[:, 0].reshape(-1,1), croped_vertices[:, 2].reshape(-1,1), croped_vertices[:,3].reshape(-1,1)), axis=1)
            if format in ['pcd', 'ply']:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(croped_vertices[:,:3]) # no intesity include here
                o3d.io.write_point_cloud(os.path.join(output_path,f"{frame_num}.{format}"), pcd)
            elif format == 'bin':
                croped_vertices.astype(np.float32).tofile(os.path.join(output_path,f"{frame_num}.bin"))

def visulaize_frames(temp_frame_queue):
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
    while not temp_frame_queue.empty:
        frame_params = temp_frame_queue.get(timeout=1)  # Timeout is optional
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
            #time.sleep(0.01)
    vis.destroy_window()
    print("The End:)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_path', type=str, required=True, help='path to the lidar record')
    parser.add_argument('--num_of_frames', type=int, default=-1,help = 'number of frames to process, -1 for all frames. default is -1')
    parser.add_argument('--format', type=str, default='pcd', choices=['bin', 'pcd', 'ply'], help='output format, default is pcd')
    parser.add_argument('--save_path', type=str, default='./dataset', help='path to save the frames')
    parser.add_argument('--normalize', type=bool, default=False, help='normalize the point cloud to [0,1] range default is False')
    parser.add_argument('--x_min', type=float, default=3, help='x min value to crop point cloud default=3 meters')
    parser.add_argument('--x_max', type=float, default=15, help='x max value to crop point cloud default=15 meters')
    parser.add_argument('--y_min', type=float, default=-10, help='y min value to crop point cloud default=-10 meters')
    parser.add_argument('--y_max', type=float, default=10, help='y max value to crop point cloud default=10 meters')
    parser.add_argument('--z_min', type=float, default=-1, help='z min value to crop point cloud default=-1 meters')
    parser.add_argument('--z_max', type=float, default=3, help='z max value to crop point cloud default=3 meters')
    args = parser.parse_args()
    pcd_cropper = PointCloudCropper(args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)
    # Create a queue
    frame_queue = queue.Queue()
    #process the data
    split_lidar_record_to_frames(args.record_path, args.num_of_frames, args.format, args.save_path, pcd_cropper, frame_queue, args.normalize, )
    #visualize
    visulaize_frames(frame_queue)
