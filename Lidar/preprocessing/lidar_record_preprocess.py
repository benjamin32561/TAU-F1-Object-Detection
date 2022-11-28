from innopy.api import FileReader, FrameDataAttributes, GrabType
import open3d as o3d
import argparse


def split_lidar_record_to_frames(record_path,num_of_frames):
    attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0)]
    frames = FileReader(record_path)
    number_of_frames = min(num_of_frames, frames.num_of_frames) if num_of_frames != -1 else frames.num_of_frames
    for i in range(number_of_frames):
        frame = frames.get_frame(i, attr)
        if frame.success:
            mes = frame.results['GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0']
            frame_num = frame.frame_number
            with open(f"lidar_record_frame{frame_num}.ply", 'w') as f:
                f.write(f"""ply
format ascii 1.0
element vertex {len(mes)}
property float32 x
property float32 y
property float32 z
property float32 distance
property int32 intensity
property int32 confidence
end_header""")
                for pixel_num in range(len(mes)):
                    f.write(f"\n{mes['x'][pixel_num]} {mes['y'][pixel_num]} {mes['z'][pixel_num]} {mes['distance'][i]} {mes['reflectivity'][i]} {mes['confidence'][i]}")
            pcd = o3d.io.read_point_cloud(f"lidar_record_frame{frame_num}.ply")
            o3d.io.write_point_cloud(f"lidar_record_frame{frame_num}.pcd", pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_path', type=str, required=True)
    parser.add_argument('--num_of_frames', type=int, default=-1)
    args = parser.parse_args()
    split_lidar_record_to_frames(args.record_path, args.num_of_frames)
