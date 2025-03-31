import numpy as np
import open3d as o3d
import roma
import torch
import cv2

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

    intrinsics = [f_x, f_y, W, H]

    # Compute 3D coordinates in camera frame
    Z = -depth_map # Negative sign due to camera coordinate convention
    X = -x * Z
    Y = y * Z

    # Use a stack to form the point cloud, of size (HxW,3)
    point_cloud = np.stack([X, Y, Z], axis=-1).reshape(-1,3)

    return point_cloud, intrinsics

def quat_to_4x4_homo(pos, quat):

    quat = torch.tensor(quat, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)

    rot = roma.unitquat_to_rotmat(quat) # This uses the xyzw convention, as is the get_pos_quat function

    homo_0 = torch.eye(4, dtype=torch.float32)  # Initialize as identity matrix
    homo_0[:3, :3] = rot  # Set the rotation part
    homo_0[:3, 3] = pos   # Set the translation part

    homo_0 = np.array(homo_0)

    return homo_0

def transform_to_world(T_cam_world, point_cloud):

    point_cloud_extended = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
    
    transformed_point_cloud = point_cloud_extended @ T_cam_world.T

    transformed_point_cloud = transformed_point_cloud[:,:3]

    return transformed_point_cloud

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

def project_to_image(point_cloud_1, intrinsics_0, T_cam0_world=None, pos_0=None, rot_0=None, camera_test=None):

    if pos_0 is not None and rot_0 is not None:

        quat_inv = roma.quat_inverse(torch.tensor(rot_0))
        rotvec_inv = np.array(roma.unitquat_to_rotvec(quat_inv))

        rotmat_inv = roma.unitquat_to_rotmat(quat_inv)
        translation_inv = np.array(-rotmat_inv.T @ pos_0)

        f_x, f_y, W, H = intrinsics_0
        K = np.zeros([3,3])
        K[0, 0], K[1, 1], K[0,2], K[1,2] = f_x, f_y, W/2, H/2

        pixel_points, _ = cv2.projectPoints(
            objectPoints=point_cloud_1.astype(np.float32), 
            rvec=rotvec_inv, 
            tvec=translation_inv, 
            cameraMatrix=K.astype(np.float32), 
            distCoeffs=np.array([])
        )

        pixel_points = pixel_points.squeeze(axis=1)  # Removes the middle dimension

        return pixel_points
    
    elif T_cam0_world is not None:

        # Calculate transformation from world back to cam0's frame
        T_world_cam0 = np.linalg.inv(T_cam0_world)
        
        rotation = torch.tensor(T_world_cam0[:3,:3])
        rotvec = roma.rotmat_to_rotvec(rotation)
        rotvec = np.array(rotvec)
        translation = T_world_cam0[:3,3]

        f_x, f_y, W, H = intrinsics_0
        K = np.zeros([3,3])
        K[0, 0], K[1, 1], K[0,2], K[1,2] = f_x, f_y, W/2, H/2

        pixel_points, _ = cv2.projectPoints(
            objectPoints=point_cloud_1.astype(np.float32), 
            rvec=rotvec, 
            tvec=translation, 
            cameraMatrix=K.astype(np.float32), 
            distCoeffs=np.array([])
        )

        pixel_points = pixel_points.squeeze(axis=1)  # Removes the middle dimension

        return pixel_points








    # Calculate transformation from world back to cam0's frame
    T_world_cam0 = np.linalg.inv(T_cam0_world)

    # Move points from world into cam0's frame
    point_cloud_extended = np.concatenate((point_cloud_1, np.ones((point_cloud_1.shape[0], 1))), axis=1)
    transformed_point_cloud = point_cloud_extended @ T_world_cam0.T

    #transformed_point_cloud = transformed_point_cloud[:,:3]
    #visualise_cams_clouds(transformed_point_cloud, camera_test)
    #breakpoint()

    # Project into ideal camera via a vanilla perspective transformation
    vanilla = np.zeros([3,4])
    vanilla[0:3, 0:3] = np.diag([1,1,1])

    pixel_coords = transformed_point_cloud @ vanilla.T

    # Map the ideal image into the real image using intrinsic matrix
    f_x, f_y, W, H = intrinsics_0
    K = np.zeros([3,3])
    K[0, 0], K[1, 1], K[0,2], K[1,2] = f_x, f_y, W/2, H/2

    pixel_coords = pixel_coords @ K.T

    # Normalize homogeneous coordinates (divide by depth)
    pixel_coords[:, 0] /= transformed_point_cloud[:, 2]  # x / z
    pixel_coords[:, 1] /= transformed_point_cloud[:, 2]  # y / z

    # Recover coordinates by removing the third column
    pixel_coords = pixel_coords[:,:2]

    return pixel_coords

def exclude_out_of_frame(ps_0, ps_1, intrinsics_0):

    _, _, W, H = intrinsics_0

    mask_0 = (
        (ps_0[:, 0] >= 0) & (ps_0[:, 0] < W) &  # Within image width
        (ps_0[:, 1] >= 0) & (ps_0[:, 1] < H) #&  # Within image height
        #(transformed_point_cloud[:, 2] > 0)  # In front of camera
    )
    ps_0 = ps_0[mask_0]
    ps_1 = ps_1[mask_0]

    mask_1 = (
        (ps_1[:, 0] >= 0) & (ps_1[:, 0] < W) &  # Within image width
        (ps_1[:, 1] >= 0) & (ps_1[:, 1] < H) #&  # Within image height
        #(transformed_point_cloud[:, 2] > 0)  # In front of camera
    )
    ps_0 = ps_0[mask_1]
    ps_1 = ps_1[mask_1]

    return ps_0, ps_1