"""
Computer Vision Coding Challenge
-------------------------------
Task:
Find pixel correspondences between two images using depth information. For each
pixel in image A, determine if it is visible in image B and if so, find its
corresponding pixel coordinates in image B.

Expected output:
Two arrays of shape (N, 2) containing the coordinates of corresponding pixels
in both images. Pixels should be color-coded based on their position in image A
for visualization (this is handled by the provided visualization code).

The dataset provides:
- RGB images
- Depth maps (in meters)
- Camera parameters (position, rotation, field of view)

Helper functions are provided for:
- Loading images and depth maps
- Accessing camera parameters
- Visualizing camera poses

The camera coordinate system is right-handed with:
- X pointing right (red axis)
- Y pointing up (green axis)
- Z pointing into the scene (negative blue axis)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import open3d as o3d
import pandas as pd

from functions import depth_map_to_pcl, transform_to_world, project_to_image, quat_to_4x4_homo, transform_to_world_new
import rerun as rr
import roma
import torch

def load_rgb_depth(image_id):
    """Load RGB image and depth map for a given image ID.

    Args:
        image_id (int): Index of the image to load

    Returns:
        tuple:
            - rgb_img (np.ndarray): RGB image of shape (H, W, 3) with values in [0, 255]
            - depth_raw (np.ndarray): Depth map of shape (H, W) containing metric distances
              from the camera for each pixel. Units are in meters.

    Note:
        The depth map is converted from 16-bit integers to metric distances using the
        min_depth and max_depth metadata stored in the PNG file.
    """
    rgb_img = Image.open(Path("dataset") / "rgb" / f"{image_id:02d}.jpg")

    depth = Image.open(Path("dataset") / "depth" / f"{image_id:02d}.png")
    min_depth = float(depth.text["min_depth"])
    max_depth = float(depth.text["max_depth"])
    depth_raw = min_depth + (
        np.asarray(depth, dtype=np.float32) * ((max_depth - min_depth) / 65535)
    )

    return np.asarray(rgb_img), depth_raw


def get_pos_rot(meta):
    """Extract camera position and rotation from metadata.

    Args:
        meta (pd.Series): Row from the metadata CSV containing camera parameters

    Returns:
        tuple:
            - pos (np.ndarray): Camera position in world coordinates (x, y, z)
            - rot (np.ndarray): Camera rotation as unit quaternion (x, y, z, w)
              representing rotation from camera to world coordinates

    Note:
        The camera coordinate system is right-handed with:
        - x-axis pointing right (red)
        - y-axis pointing up (green)
        - z-axis pointing backwards into the scene (negative blue)
    """
    pos = meta[["pos_x", "pos_y", "pos_z"]].values.astype(np.float32)
    rot = meta[["quat_x", "quat_y", "quat_z", "quat_w"]].values.astype(np.float32)
    return pos, rot


def create_camera_gizmo(
    t_cam=np.eye(4), fov_h_deg=120, img_shape=(1, 1), frustum_distance=0.25
):
    """Create a visualization of a camera's position, orientation and field of view.

    Args:
        t_cam (np.ndarray): 4x4 homogeneous transformation matrix from camera to world coordinates
        fov_h_deg (float): Horizontal field of view in degrees
        img_shape (tuple): Image dimensions as (height, width)
        frustum_distance (float): Distance to render the camera frustum. Set to None to disable

    Returns:
        list: List of Open3D geometry objects representing:
            - Coordinate axes (RGB for XYZ)
            - Camera frustum (black wireframe) if frustum_distance is not None

    Note:
        - The coordinate axes show the camera's local coordinate system
        - The frustum visualizes the camera's field of view as a pyramid
        - Use with o3d.visualization.draw_geometries([*create_camera_gizmo(...)])
    """
    # Create empty line set for coordinate axes
    coord_axes = o3d.geometry.LineSet()

    # Origin point
    axis_origin = (t_cam @ np.array([0, 0, 0, 1]))[:3]

    # Create coordinate axes points and lines
    ax_len = 0.5
    axes_points = [
        axis_origin,
        (t_cam @ np.array([ax_len, 0, 0, 1]))[:3],  # X axis
        (t_cam @ np.array([0, ax_len, 0, 1]))[:3],  # Y axis
        (t_cam @ np.array([0, 0, ax_len, 1]))[:3],  # Z axis
    ]

    axes_lines = [[0, 1], [0, 2], [0, 3]]  # Connect origin to each axis end
    axes_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB colors

    coord_axes.points = o3d.utility.Vector3dVector(axes_points)
    coord_axes.lines = o3d.utility.Vector2iVector(axes_lines)
    coord_axes.colors = o3d.utility.Vector3dVector(axes_colors)

    geometries = [coord_axes]

    # Add frustum if requested
    if frustum_distance is not None and frustum_distance > 0.0:
        frustum = o3d.geometry.LineSet()

        fov_h_rad = np.deg2rad(fov_h_deg)
        span_h = frustum_distance * np.tan(fov_h_rad / 2)
        h, w = img_shape
        aspect_ratio = w / h
        span_w = span_h / aspect_ratio

        # Create frustum points
        frustum_points = [
            axis_origin,
            (t_cam @ np.array([-span_h, span_w, -frustum_distance, 1.0]))[:3],
            (t_cam @ np.array([span_h, span_w, -frustum_distance, 1.0]))[:3],
            (t_cam @ np.array([span_h, -span_w, -frustum_distance, 1.0]))[:3],
            (t_cam @ np.array([-span_h, -span_w, -frustum_distance, 1.0]))[:3],
        ]

        # Create lines for frustum edges
        frustum_lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # Lines from origin to corners
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],  # Lines connecting corners
        ]

        # Black color for all frustum lines
        frustum_colors = [[0, 0, 0] for _ in range(len(frustum_lines))]

        frustum.points = o3d.utility.Vector3dVector(frustum_points)
        frustum.lines = o3d.utility.Vector2iVector(frustum_lines)
        frustum.colors = o3d.utility.Vector3dVector(frustum_colors)

        geometries.append(frustum)

    return geometries


def visualize(filename, img_0, img_1, ps_0, ps_1):
    """Visualize corresponding pixels between two images.
    Points are color-coded based on their polar coordinates in image 0."""

    # Calculate center of source image
    h, w = img_0.shape[:2]
    center_y, center_x = h / 2, w / 2

    # Calculate angles for hue
    angles = np.arctan2(ps_0[:, 1] - center_y, ps_0[:, 0] - center_x)
    hues = (angles + np.pi) / (2 * np.pi)

    fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(10, 5))
    for ax in axs.flatten():
        ax.axis("off")
    axs[0].imshow(img_0)
    axs[0].scatter(ps_0[:, 0], ps_0[:, 1], c=hues, cmap="hsv", s=0.2, alpha=0.3)
    axs[1].imshow(img_1)
    axs[1].scatter(ps_1[:, 0], ps_1[:, 1], c=hues, cmap="hsv", s=0.2, alpha=0.3)
    plt.savefig(filename, dpi=300)
    plt.show()


def generate(idx):
    meta = pd.read_csv("dataset/data.csv")
    meta_0 = meta.iloc[idx[0]]
    meta_1 = meta.iloc[idx[1]]

    img_0, depth_0 = load_rgb_depth(idx[0])
    img_1, depth_1 = load_rgb_depth(idx[1])
    # depth is now metric distance for each pixel

    fig, axs = plt.subplots(2, 2, layout="constrained", figsize=(8, 4))
    axs[0][0].imshow(img_0)
    axs[0][1].imshow(depth_0)
    axs[1][0].imshow(img_1)
    axs[1][1].imshow(depth_1)
    #plt.show()

    pos_0, rot_0 = get_pos_rot(meta_0)
    pos_1, rot_1 = get_pos_rot(meta_1)
    # pos is x, y, z; rot is quaternion x, y, z, w

    homo_0 = quat_to_4x4_homo(pos_0, rot_0) # The problem lies here!
    pcl_0 = depth_map_to_pcl(depth_0, meta_0["cam_fov"])
    transformed_pcl_0 = transform_to_world_new(homo_0, pcl_0)
    camera_0 = create_camera_gizmo(homo_0, meta_0.cam_fov, img_0.shape[:2], 0.25)

    homo_1 = quat_to_4x4_homo(pos_1, rot_1)
    pcl_1 = depth_map_to_pcl(depth_1, meta_1["cam_fov"])
    transformed_pcl_1 = transform_to_world_new(homo_1, pcl_1)
    camera_1 = create_camera_gizmo(homo_1, meta_1.cam_fov, img_1.shape[:2], 0.25)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(transformed_pcl_0)
    pcd0.paint_uniform_color([1,0,0])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(transformed_pcl_1)
    pcd1.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pcd0] + [pcd1] + camera_0 + camera_1)


    return
































    homo_0 = quat_to_4x4_homo(pos_0, rot_0)
    homo_1 = quat_to_4x4_homo(pos_1, rot_1)

    # Sample code to show both camera gizmo at origin
    camera_0 = create_camera_gizmo(homo_0, meta_0.cam_fov, img_0.shape[:2], 0.25)
    camera_1 = create_camera_gizmo(homo_1, meta_1.cam_fov, img_0.shape[:2], 0.25)

    # TODO: Implement your solution here
    # Expected steps:

    # 1. Convert depth maps to point clouds

    pcl_0 = depth_map_to_pcl(depth_0, meta_0["cam_fov"])
    pcl_1 = depth_map_to_pcl(depth_1, meta_1["cam_fov"])

    #rr.init("depth_point_cloud", spawn=True)
    #rr.log("point_cloud", rr.Points3D(visualisable_pcl1, colors=[(0, 255, 0)] * len(visualisable_pcl1)))

    # 2. Transform points between camera coordinate systems

    transformed_pcl_0 = transform_to_world(pcl_0, pos_0, rot_0)
    transformed_pcl_1 = transform_to_world(pcl_1, pos_1, rot_1)

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(transformed_pcl_0)
    pcd0.paint_uniform_color([1,0,0])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(transformed_pcl_1)
    pcd1.paint_uniform_color([0,1,0])

    # Visualize
    o3d.visualization.draw_geometries([pcd0] + [pcd1] + camera_0 + camera_1)

    breakpoint()

    rr.init("depth_point_cloud", spawn=True)
    rr.log("point_cloud_0", rr.Points3D(transformed_pcl_0, colors=(0, 255, 0)))
    rr.log("point_cloud_1", rr.Points3D(transformed_pcl_1, colors=(255, 0, 0)))
    breakpoint()

    # 3. Project points into image space

    points1 = project_to_image()
    points2 = project_to_image()

    # 4. Find and return corresponding pixels ps_0 of shape (N, 2) and ps_1 of shape (N, 2)


    ps_0 = (np.random.rand(10000, 2) * 224).astype(np.int32)
    ps_1 = (np.random.rand(10000, 2) * 224).astype(np.int32)
    return img_0, ps_0, img_1, ps_1


if __name__ == "__main__":
    sample_idxs = [
        [2, 4],
        [6, 9],
        [3, 5],
        [0, 1],
        [7, 8],
    ]
    for i, sample_idx in enumerate(sample_idxs):
        img_0, ps_0, img_1, ps_1 = generate(sample_idx)
        visualize(f"result_{i}.jpg", img_0, img_1, ps_0, ps_1)
