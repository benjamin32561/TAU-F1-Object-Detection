import numpy as np
from innopy.api import FileReader, FrameDataAttributes, GrabType
from sklearn.preprocessing import minmax_scale
import open3d as o3d
import argparse
import os


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


def split_lidar_record_to_frames(record_path,num_of_frames, format, output_path,pcd_cropper, normalize=False):
    attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0),FrameDataAttributes(GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA)]
    config_path = '../recording_remove_blooming_config.json'
    frames = FileReader(record_path, num_of_cores=1,config_filepath=config_path)
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
            croped_vertices = pcd_cropper(vertices_np)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_path', type=str, required=True)
    parser.add_argument('--num_of_frames', type=int, default=-1)
    parser.add_argument('--format', type=str, default='pcd', choices=['bin', 'pcd', 'ply'])
    parser.add_argument('--save_path', type=str, default='./dataset')
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--x_min', type=float, default=3, help='x min value to crop point cloud default=3 meters')
    parser.add_argument('--x_max', type=float, default=15, help='x max value to crop point cloud default=15 meters')
    parser.add_argument('--y_min', type=float, default=-10, help='y min value to crop point cloud default=-10 meters')
    parser.add_argument('--y_max', type=float, default=10, help='y max value to crop point cloud default=10 meters')
    parser.add_argument('--z_min', type=float, default=-1, help='z min value to crop point cloud default=-1 meters')
    parser.add_argument('--z_max', type=float, default=3, help='z max value to crop point cloud default=3 meters')
    args = parser.parse_args()
    pcd_cropper = PointCloudCropper(args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)
    split_lidar_record_to_frames(args.record_path, args.num_of_frames, args.format, args.save_path, pcd_cropper, args.normalize, )
