import numpy as np
import open3d as o3d
import roma
import torch

def depth_map_to_pcl(depth_map, cam_fov):

    # Infer camera intrinsics using pinhole model
    H, W = depth_map.shape
    fov_h_rad = np.deg2rad(cam_fov)
    f_x = W / (2 * np.tan(fov_h_rad / 2))
    f_y = f_x
    c_x, c_y = W / 2, H / 2

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Convert to normalized coordinates
    x = (u - c_x) / f_x
    y = (v - c_y) / f_y

    # Compute 3D coordinates in camera frame
    Z = -depth_map # Negative sign due to camera coordinate convention
    X = -x * Z
    Y = y * Z

    # Use a stack to form the point cloud, of size (HxW,3)
    point_cloud = np.stack([X, Y, Z], axis=-1).reshape(-1,3)

    return point_cloud

def transform_to_world(point_cloud, pos, unitquat):

    # Convert to tensors, explicitly giving precision
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
    unitquat = torch.tensor(unitquat, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)

    # Convert quaternion to rotation matrix
    unitquat = roma.quat_wxyz_to_xyzw(unitquat)
    rotmat = roma.unitquat_to_rotmat(unitquat)

    # Perform the transformation
    point_cloud_world = (rotmat @ point_cloud.T).T + pos

    return point_cloud_world

def project_to_image():



    pass

def quat_to_4x4_homo(pos, quat):

    quat = torch.tensor(quat, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)

    rot = roma.unitquat_to_rotmat(quat) # This uses the xyzw convention, as is the get_pos_quat function

    homo_0 = torch.eye(4, dtype=torch.float32)  # Initialize as identity matrix
    homo_0[:3, :3] = rot  # Set the rotation part
    homo_0[:3, 3] = pos   # Set the translation part

    homo_0 = np.array(homo_0)

    return homo_0

def transform_to_world_new(homo, point_cloud):

    point_cloud_extended = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
    
    transformed_point_cloud = point_cloud_extended @ homo.T

    transformed_point_cloud = transformed_point_cloud[:,:3]

    return transformed_point_cloud