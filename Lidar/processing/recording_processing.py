import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import random
from time import sleep
import open3d as o3d
import asyncio
import numpy as np
from classes import PointCloudCropper, FrameParameters
from innopy.api import FileReader, FrameDataAttributes, GrabType
target_path = "/home/idola/PycharmProjects/patchwork-plusplus/build/python_wrapper"

try:
    patchwork_module_path = target_path
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

DEFAULT_ATTRS = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0),
                      FrameDataAttributes(GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA)]


class PointCloudPatchworkPipeline:

    def __init__(self, config_path, attributes=DEFAULT_ATTRS,
                 sensor_height=0.8, elevation_thr=[-0.8, -0.2, 0.2, 0.8],
                 x_min=3, x_max=15, y_max=6, y_min=-6, z_max=3, z_min=-1, verbose=False):
        self.config_path = config_path
        self.attributes = attributes
        self.pcc = PointCloudCropper(x_min=x_min, x_max=x_max, y_max=y_max, y_min=y_min, z_max=z_max, z_min=z_min)
        self.patchwork = self.init_patchwork(sensor_height, elevation_thr, verbose)

    @staticmethod
    def init_patchwork(sensor_height, elevation_thr, verbose):
        params = pypatchworkpp.Parameters()
        params.verbose = verbose
        params.sensor_height = sensor_height
        params.elevation_thr = elevation_thr
        return pypatchworkpp.patchworkpp(params)

    def process_recording(self, file_path, max_frames=np.inf):
        frames = FileReader(file_path, num_of_cores=1, config_filepath=self.config_path)
        for i in range(int(min(max_frames, frames.num_of_frames))):
            frame, frame_meta = self._get_frame(frames, i)
            if frame is not None:
                vertices = self._get_vertices(frame, frame_meta)
                cropped_vertices = self.pcc(vertices)
                self.patchwork.estimateGround(cropped_vertices)
                ground = self.patchwork.getGround()
                non_ground = self.patchwork.getNonground()
                frame_patch_map = self._get_patch_map(non_ground)
                frame_objects_map = self._get_object_map(frame_patch_map, threshold=5)  # need to think about threshold's value
                time_taken = self.patchwork.getTimeTaken()
                patchwork_results = FrameParameters(ground, non_ground, frame_objects_map, time_taken)
                yield patchwork_results

    def _get_frame(self, frames, frame_num):
        frame = frames.get_frame(frame_num, self.attributes)
        if frame.success:
            return frame.results['GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0'], frame.results['GrabType.GRAB_TYPE_SINGLE_PIXEL_META_DATA']
        return None, None

    @staticmethod
    def _get_vertices(frame, frame_meta):
        vertices = []
        for pixel_num in range(len(frame)):
            if frame['confidence'][pixel_num] > 0 and frame_meta['ghost'][pixel_num] == 0 and frame_meta['noise'][pixel_num] == 0:
                vertices.append([frame['x'][pixel_num], frame['y'][pixel_num], frame['z'][pixel_num], frame['reflectivity'][pixel_num]])
        return np.asarray(vertices)/100

    def _get_patch_map(self, non_ground):
        # Use a dictionary to store unique pixels based on a unique identifier
        unique_patches = {}
        # Iterate through each pixel and use a unique combination of distance, ring, and sector as a key
        for pixel in non_ground:
            patch = tuple(pixel[-3:])  # 3: ring ; 4: sector ; 5: zone
            if patch in unique_patches.keys():
                unique_patches[patch].append(pixel[:3])
            else:
                unique_patches[patch] = [pixel[:3]]
        return unique_patches

    def _get_object_map(self, hash, threshold):
        object_table = {}
        for patch in hash.keys():
            if len(hash[patch]) > threshold:
                object_table[patch] = hash[patch]  # copy only objects
        return object_table


class PointCloudVisualizer:
    def __init__(self, width=640, height=480):  # Default size is 640x480, adjust as needed
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height)
        self.ground_pcd = o3d.geometry.PointCloud()
        self.non_ground_pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.ground_pcd)
        #self.vis.add_geometry(self.non_ground_pcd)
        self.initialized = False
        self.objects_pcd = []  # List to store point clouds
        self.bounded_o3d_objects = []  # List to store bounding boxes

    def init_bbox(self, object_pixels):
        for patch in object_pixels.keys():
            object_points = object_pixels[patch]
            objects_o3d = o3d.geometry.PointCloud()
            objects_o3d.points = o3d.utility.Vector3dVector(object_points)
            random_color = [random.random(), random.random(), random.random()]
            objects_o3d.paint_uniform_color(random_color)
            self.objects_pcd.append(objects_o3d)
            bbox = objects_o3d.get_axis_aligned_bounding_box()
            bbox.color = [255, 0, 0]
            self.bounded_o3d_objects.append(bbox)
            self.vis.add_geometry(bbox)
        #for bbox, pcd in zip(self.bounded_o3d_objects, self.objects_pcd):
            self.vis.add_geometry(objects_o3d)

    def update_bbox(self, object_pixels):
        # Remove existing point clouds and bounding boxes
        for pcd, bbox in zip(self.objects_pcd, self.bounded_o3d_objects):
            self.vis.remove_geometry(pcd)
            self.vis.remove_geometry(bbox)

        # Clear the lists of point clouds and bounded objects
        self.objects_pcd.clear()
        self.bounded_o3d_objects.clear()

        # Re-initialize point clouds and bounding boxes
        for patch in object_pixels.keys():
            object_points = object_pixels[patch]
            objects_o3d = o3d.geometry.PointCloud()
            objects_o3d.points = o3d.utility.Vector3dVector(object_points)
            random_color = [random.random(), random.random(), random.random()]
            objects_o3d.paint_uniform_color(random_color)
            self.objects_pcd.append(objects_o3d)
            # Recalculate the bounding box based on the updated point cloud data
            bbox = objects_o3d.get_axis_aligned_bounding_box()
            bbox.color = [255, 0, 0]
            #print("max bound = ", max_bound, "\n")
            self.vis.add_geometry(bbox)
            # Add the updated point cloud and bounding box to the visualizer
            self.vis.add_geometry(objects_o3d)
            # Add the bounding box to the list of bounded objects
            self.bounded_o3d_objects.append(bbox)

    def run(self, processed_frames_queue):
        while True:
            try:
                patchwork_results = processed_frames_queue.get(timeout=1)
                if patchwork_results is None:
                    break
                self.update_visualization(patchwork_results)
            except queue.Empty:
                sleep(1)
                continue
            self.vis.poll_events()

    def update_visualization(self, patchwork_results):
        # Update point cloud data
        self.ground_pcd.points = o3d.utility.Vector3dVector(patchwork_results.ground[:, :3])
        self.non_ground_pcd.points = o3d.utility.Vector3dVector(patchwork_results.non_ground[:, :3])

        # Update colors if available and needed
        if patchwork_results.ground.shape[1] > 3:
            self.ground_pcd.colors = o3d.utility.Vector3dVector(
                np.array([[0.0, 1.0, 0.0] for _ in range(patchwork_results.ground[:, 3:].shape[0])], dtype=float))  # RGB patchwork_results.ground[:, 3:])

        if patchwork_results.non_ground.shape[1] > 3:
            self.non_ground_pcd.colors = o3d.utility.Vector3dVector(patchwork_results.non_ground[:, 3:])

        # Only add the geometries to the visualizer once
        if not self.initialized:
            self.vis.add_geometry(self.ground_pcd)
            #self.vis.add_geometry(self.non_ground_pcd)
            self.init_bbox(patchwork_results.object_pixels)
            self.initialized = True

        # This is all you need to update the visualization
        self.vis.update_geometry(self.ground_pcd)
        #self.vis.update_geometry(self.non_ground_pcd)
        self.update_bbox(patchwork_results.object_pixels)
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


def process_point_clouds(file_path, config_path, processed_frames_queue):
    pipeline = PointCloudPatchworkPipeline(config_path)
    for patchwork_results in pipeline.process_recording(file_path):
        processed_frames_queue.put(patchwork_results)
    processed_frames_queue.put(None)  # Signal that processing is complete


def main():
    file_path = '/home/idola/PycharmProjects/TAU-F1-Object-Detection/Lidar/processing/Recordings'
    config_path = '/home/idola/PycharmProjects/TAU-F1-Object-Detection/Lidar/innoviz_api/examples/lidar_configuration_files/recording_remove_blooming_config.json'
    processed_frames_queue = queue.Queue(maxsize=20)

    processing_thread = threading.Thread(target=process_point_clouds,
                                         args=(file_path, config_path, processed_frames_queue))
    processing_thread.start()
    sleep(5)
    visualizer = PointCloudVisualizer()
    try:
        visualizer.run(processed_frames_queue)
    finally:
        visualizer.close()

    processing_thread.join()


if __name__ == "__main__":
    main()