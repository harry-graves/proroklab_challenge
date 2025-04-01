from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np
import torch
import roma
import cv2

def depth_map_to_pcl(depth_map, cam_fov):
    """
    Converts a depth map to a point cloud using PyTorch.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        cam_fov (float): Camera field of view in degrees.

    Returns:
        torch.Tensor: Point cloud of shape (H*W, 3).
        torch.Tensor: Intrinsic parameters [f_x, f_y, W, H].
    """
    # Infer camera intrinsics using the pinhole model
    H, W = depth_map.shape
    fov_h_rad = torch.deg2rad(torch.tensor(cam_fov))
    f_x = W / (2 * torch.tan(fov_h_rad / 2))
    f_y = f_x  # Assuming square images
    c_x, c_y = W / 2, H / 2

    # Store a tensor of intrinsics
    intrinsics = torch.tensor([f_x, f_y, W, H], dtype=torch.float32)

    # Create pixel coordinate grid
    u = torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(H, W)
    v = torch.arange(H, dtype=torch.float32).unsqueeze(1).expand(H, W)

    # Convert to normalized coordinates
    x = (u - c_x) / f_x
    y = (v - c_y) / f_y

    # Compute 3D coordinates in camera frame
    Z = -depth_map  # Negative sign due to camera coordinate convention
    X = -x * Z  # Negative sign due to camera coordinate convention
    Y = y * Z

    # Stack to form the point cloud, of size (H*W, 3)
    point_cloud = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    return point_cloud, intrinsics

def quat_to_4x4_homog(pos, quat):
    """
    Converts a position and unit quaternion into a 4x4 homogeneous transformation matrix.

    Args:
        pos (array-like or torch.Tensor): Translation vector of shape (3,).
        quat (array-like or torch.Tensor): Unit quaternion [x, y, z, w] of shape (4,).

    Returns:
        torch.Tensor: A 4x4 homogeneous transformation matrix.
    
    Notes:
        - Assumes the quaternion follows the (x, y, z, w) convention.
        - Uses `roma.unitquat_to_rotmat()` to convert the quaternion to a rotation matrix.
        - The output matrix can be used to transform points from one coordinate frame to another.
    """
    quat = torch.tensor(quat, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)

    rot = roma.unitquat_to_rotmat(quat) # This uses the xyzw convention, as does the get_pos_quat function

    homog = torch.eye(4, dtype=torch.float32)
    homog[:3, :3] = rot
    homog[:3, 3] = pos

    return homog

def transform_to_world(T_cam_world, point_cloud):
    """
    Transforms a point cloud from the camera frame to the world frame using a homogeneous transformation matrix.

    Args:
        T_cam_world (torch.Tensor): A 4x4 homogeneous transformation matrix representing the camera pose in the world frame.
        point_cloud (torch.Tensor): A tensor of shape (N, 3) containing 3D points in the camera frame.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing the transformed 3D points in the world frame.

    Notes:
        - The function first converts the point cloud to homogeneous coordinates by appending a column of ones.
        - It then applies the transformation using matrix multiplication.
        - The final result extracts only the (x, y, z) coordinates from the homogeneous output.
    """
    point_cloud_extended = torch.concatenate((point_cloud, torch.ones((point_cloud.shape[0], 1))), axis=1)
    
    transformed_point_cloud = point_cloud_extended @ T_cam_world.T

    transformed_point_cloud = transformed_point_cloud[:,:3]

    return transformed_point_cloud

def find_common_points(pcl_0, pcl_1, threshold=0.01):
    """
    Identifies common points between two point clouds by finding nearest neighbors within a specified threshold.

    Args:
        pcl_0 (numpy.ndarray): A point cloud of shape (N, 3).
        pcl_1 (numpy.ndarray): A second point cloud of shape (M, 3).
        threshold (float, optional): The maximum allowed distance between matched points. Defaults to 0.01.

    Returns:
        numpy.ndarray: A filtered point cloud containing only the points from `pcl_0` that have a match in `pcl_1` within the threshold.

    Notes:
        - Uses a KDTree for fast nearest-neighbor search.
        - Points in `pcl_0` are retained if they have a neighbor in `pcl_1` within the given threshold.
        - This method assumes both point clouds are already aligned in the same coordinate frame.
    """
    tree = cKDTree(pcl_1)
    distances, _ = tree.query(pcl_0, distance_upper_bound=threshold)

    mask = distances < threshold
    return pcl_0[mask]

def project_to_image(point_cloud, intrinsics, T_cam_world):
    """
    Projects a 3D point cloud into 2D image coordinates using camera intrinsics and extrinsics.

    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 3) representing 3D points in the world coordinate frame.
        intrinsics (list or tuple): Camera intrinsics [f_x, f_y, W, H], where:
            - f_x, f_y: Focal lengths in pixels.
            - W, H: Image width and height.
        T_cam_world (torch.Tensor): A 4x4 homogeneous transformation matrix representing the camera pose in the world frame.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the 2D pixel coordinates of the projected points.

    Notes:
        - Converts the world-to-camera transformation to OpenCV format.
        - Uses Rodrigues' rotation formula to convert the rotation matrix to a rotation vector.
        - Applies OpenCV's `cv2.projectPoints` for perspective projection.
        - Corrects the x-coordinates to account for OpenCV’s coordinate convention.
        - Assumes no lens distortion (distCoeffs set to an empty array).
    """
    # Calculate transformation from world back to cam0's frame
    T_world_cam0 = torch.linalg.inv(T_cam_world)
    
    # Put everything into OpenCV format
    rotation = T_world_cam0[:3,:3]
    rotvec = roma.rotmat_to_rotvec(rotation)
    translation = T_world_cam0[:3,3]

    # Define intrinsic matrix K
    f_x, f_y, W, H = intrinsics
    K = torch.zeros([3,3])
    K[0, 0], K[1, 1], K[0,2], K[1,2] = f_x, f_y, W/2, H/2

    # Project points
    pixel_points, _ = cv2.projectPoints(
        objectPoints=np.array(point_cloud, dtype=np.float32), 
        rvec=np.array(rotvec, dtype=np.float32), 
        tvec=np.array(translation, dtype=np.float32), 
        cameraMatrix=np.array(K, dtype=np.float32), 
        distCoeffs=np.array([])
    )

    # Convert back to a PyTorch tensor
    pixel_points = torch.tensor(pixel_points)

    # Remove the middle dimension
    pixel_points = pixel_points.squeeze(axis=1)

    # Flip x coords due to difference in OpenCV's coordinate convention
    pixel_points[:,0] = W - pixel_points[:,0]

    return pixel_points

def remove_border_points(ps_0, ps_1, intrinsics_0, intrinsics_1):
    """
    Filters out points that fall outside the valid image boundaries in both camera views.

    Args:
        ps_0 (torch.Tensor): A tensor of shape (N, 2) representing pixel coordinates in the first image.
        ps_1 (torch.Tensor): A tensor of shape (N, 2) representing pixel coordinates in the second image.
        intrinsics_0 (list or tuple): Camera intrinsics [f_x, f_y, W, H] for the first camera.
        intrinsics_1 (list or tuple): Camera intrinsics [f_x, f_y, W, H] for the second camera.

    Returns:
        tuple:
            - ps_0 (torch.Tensor): Filtered pixel coordinates for the first image.
            - ps_1 (torch.Tensor): Filtered pixel coordinates for the second image.

    Notes:
        - First, points in `ps_0` are filtered based on the image size of the first camera.
        - Corresponding points in `ps_1` are also filtered accordingly.
        - The process is then repeated for `ps_1`, ensuring that only points within both image boundaries remain.
    """
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
    """
    Visualizes one or two point clouds along with their corresponding camera coordinate frames 
    using Open3D.

    Args:
        point_cloud_0 (numpy.ndarray or torch.Tensor, optional): 
            The first point cloud of shape (N, 3). Colored red.
        camera_0 (list of o3d.geometry.Geometry, optional): 
            A list of Open3D geometries representing the first camera's coordinate frame.
        point_cloud_1 (numpy.ndarray or torch.Tensor, optional): 
            The second point cloud of shape (M, 3). Colored green.
        camera_1 (list of o3d.geometry.Geometry, optional): 
            A list of Open3D geometries representing the second camera's coordinate frame.

    Behavior:
        - If `point_cloud_1` and `camera_1` are provided, both point clouds and cameras are displayed.
        - If only `point_cloud_0` and `camera_0` are provided, only the first set is displayed.
        - The first point cloud is colored red, and the second is colored green.

    Returns:
        None. The function launches an interactive Open3D visualization window.

    Notes:
        - The function assumes that the camera coordinate frames are precomputed Open3D geometries.
        - Point clouds are converted into Open3D format before visualization.
    """
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