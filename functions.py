from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np
import torch
import roma
import cv2

def depth_map_to_pcl(depth_map, cam_fov):

    # Infer camera intrinsics using pinhole model
    H, W = depth_map.shape
    fov_h_rad = np.deg2rad(cam_fov)
    f_x = W / (2 * np.tan(fov_h_rad / 2))
    f_y = f_x # Assuming all images are square, which at this stage they are
    c_x, c_y = W / 2, H / 2

    # Store a list of intrinsics for downstream tasks
    intrinsics = [f_x, f_y, W, H]

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Convert to normalized coordinates
    x = (u - c_x) / f_x
    y = (v - c_y) / f_y

    # Compute 3D coordinates in camera frame
    Z = -depth_map # Negative sign due to camera coordinate convention
    X = -x * Z # Negative sign due to camera coordinate convention
    Y = y * Z

    # Use a stack to form the point cloud, of size (HxW,3)
    point_cloud = np.stack([X, Y, Z], axis=-1).reshape(-1,3)

    return point_cloud, intrinsics

def quat_to_4x4_homog(pos, quat):

    quat = torch.tensor(quat, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)

    rot = roma.unitquat_to_rotmat(quat) # This uses the xyzw convention, as does the get_pos_quat function

    homog = torch.eye(4, dtype=torch.float32)
    homog[:3, :3] = rot
    homog[:3, 3] = pos

    homog = np.array(homog)

    return homog

def transform_to_world(T_cam_world, point_cloud):

    point_cloud_extended = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
    
    transformed_point_cloud = point_cloud_extended @ T_cam_world.T

    transformed_point_cloud = transformed_point_cloud[:,:3]

    return transformed_point_cloud

def find_common_points(pcl_0, pcl_1, threshold=0.01):
    """
    Finds points that appear in both point clouds (within a given threshold).

    Args:
        pc1 (np.ndarray): First point cloud of shape (N, 3).
        pc2 (np.ndarray): Second point cloud of shape (M, 3).
        threshold (float): Distance threshold to consider points as matching.

    Returns:
        np.ndarray: Common points in the first point cloud.
    """
    tree = cKDTree(pcl_1)  # Build KDTree for efficient nearest-neighbor search
    distances, indices = tree.query(pcl_0, distance_upper_bound=threshold)

    # Keep only points where a match was found
    mask = distances < threshold
    return pcl_0[mask]

def project_to_image(point_cloud, intrinsics, T_cam_world):

    # Calculate transformation from world back to cam0's frame
    T_world_cam0 = np.linalg.inv(T_cam_world)
    
    # Put everything into OpenCV format
    rotation = torch.tensor(T_world_cam0[:3,:3])
    rotvec = roma.rotmat_to_rotvec(rotation)
    rotvec = np.array(rotvec)
    translation = T_world_cam0[:3,3]

    # Define intrinsic matrix K
    f_x, f_y, W, H = intrinsics
    K = np.zeros([3,3])
    K[0, 0], K[1, 1], K[0,2], K[1,2] = f_x, f_y, W/2, H/2

    # Project points
    pixel_points, _ = cv2.projectPoints(
        objectPoints=point_cloud.astype(np.float32), 
        rvec=rotvec, 
        tvec=translation, 
        cameraMatrix=K.astype(np.float32), 
        distCoeffs=np.array([])
    )

    # Remove the middle dimension
    pixel_points = pixel_points.squeeze(axis=1)

    # Flip x coords due to difference in OpenCV's coordinate convention
    pixel_points[:,0] = W - pixel_points[:,0]

    return pixel_points

def remove_border_points(ps_0, ps_1, intrinsics_0, intrinsics_1):

    _, _, W, H = intrinsics_0

    mask_0 = (
    (ps_0[:, 0] >= 0) & (ps_0[:, 0] < W) & # Within image width
    (ps_0[:, 1] >= 0) & (ps_0[:, 1] < H) # Within image height
    )

    ps_0 = ps_0[mask_0]
    ps_1 = ps_1[mask_0]

    _, _, W, H = intrinsics_1

    mask_1 = (
    (ps_1[:, 0] >= 0) & (ps_1[:, 0] < W) & # Within image width
    (ps_1[:, 1] >= 0) & (ps_1[:, 1] < H) # Within image height
    )

    ps_0 = ps_0[mask_1]
    ps_1 = ps_1[mask_1]

    return ps_0, ps_1

def visualise_cams_clouds(point_cloud_0=None, camera_0=None, point_cloud_1=None, camera_1=None):

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(point_cloud_0)
    pcd0.paint_uniform_color([1,0,0])

    if point_cloud_1 is None:
        o3d.visualization.draw_geometries([pcd0] + camera_0)

    else:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_cloud_1)
        pcd1.paint_uniform_color([0,1,0])

        o3d.visualization.draw_geometries([pcd0] + [pcd1] + camera_0 + camera_1)