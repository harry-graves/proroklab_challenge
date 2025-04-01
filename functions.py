import open3d as o3d
import torch
import roma

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

def project_to_image(point_cloud, intrinsics, T_cam_world):
    """
    Projects 3D points from the world coordinate frame to 2D pixel coordinates in the camera image.

    The function first transforms the 3D points from the world coordinate system to the camera's local coordinate system 
    using the inverse of the camera's pose (T_cam_world). Then, it applies the camera's intrinsic matrix to project the 
    points onto the 2D image plane. The resulting pixel coordinates are flipped horizontally to match the standard 
    image coordinate convention.

    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 3) representing 3D points in the world coordinate frame, 
                                    where N is the number of points.
        intrinsics (tuple): A tuple containing the camera's intrinsic parameters (f_x, f_y, W, H)
        T_cam_world (torch.Tensor): A 4x4 tensor representing the transformation matrix from the world coordinate 
                                    frame to the camera's coordinate frame.

    Returns:
        tuple:
            - pixel_coords (torch.Tensor): A tensor of shape (N, 2) representing the 2D pixel coordinates of the 
                                          projected points in the image, with the x and y coordinates flipped 
                                          according to the camera coordinate convention.
            - point_cloud_cam_frame (torch.Tensor): A tensor of shape (N, 3) representing the transformed 3D points 
                                                     in the camera's local coordinate frame.

    Notes:
        - The function assumes that the input point cloud is in the world coordinate system and the camera's intrinsic 
          matrix and pose are provided for a valid projection.
        - The function flips the x-coordinate of the resulting pixel coordinates to align with the typical image 
          coordinate convention, where the origin is at the top-left corner and the x-axis points to the right.
    """
    # Calculate transformation from world back to cam0's frame
    T_world_cam = torch.linalg.inv(T_cam_world)

    # Move points from world into cam0's frame
    point_cloud_extended = torch.cat((point_cloud, torch.ones((point_cloud.shape[0], 1))), dim=1)
    point_cloud_cam_frame = (point_cloud_extended @ T_world_cam.T)[:, :3]  # Extract XYZ

    # Project into ideal camera via perspective transformation
    # NOTE - this clone is redundant, but has been kept for readability
    pixel_coords = point_cloud_cam_frame.clone()  # (N, 3)

    # Map the ideal image into the real image using intrinsic matrix
    f_x, f_y, W, H = intrinsics
    K = torch.tensor([
        [f_x,  0,   W / 2],
        [ 0,  f_y,  H / 2],
        [ 0,   0,    1   ]
    ], dtype=torch.float32)

    # Apply intrinsics
    pixel_coords = (K @ pixel_coords.T).T  # (N, 3)

    # Convert from homogeneous to Cartesian by dividing by depth (Z)
    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:3]

    
    # Convert to integer pixel coordinates
    pixel_coords = torch.tensor(pixel_coords, dtype=torch.int)
    
    # Flip x coords due to difference in coordinate convention
    pixel_coords[:,0] = W - pixel_coords[:,0]

    return pixel_coords, point_cloud_cam_frame

def filter_points(ps_0, intrinsics_0, pcl_1_frame_cam0, pcl_0_frame_cam1, depth_0, depth_threshold=0.1):
    """
    Filters out points that fall outside image boundaries, are behind cameras, or are occluded.

    Args:
        ps_0 (torch.Tensor): A tensor of shape (N, 2) representing pixel coordinates in the first image.
        intrinsics_0 (tuple or list): Camera intrinsics [f_x, f_y, W, H] for the first camera.
        pcl_1_frame_cam0 (torch.Tensor): (N, 3) point cloud transformed from camera 1 to camera 0's frame.
        pcl_0_frame_cam1 (torch.Tensor): (N, 3) point cloud transformed from camera 0 to camera 1's frame.
        depth_0 (torch.Tensor): A tensor of shape (N,) representing depth values of the first camera's image.
        depth_threshold (float, optional): A threshold to account for depth errors when checking occlusion. 
                                           Defaults to 0.1.

    Returns:
        torch.Tensor: A boolean mask of shape (N,) indicating which points are valid.
    
    Notes:
        - Points outside the image width/height are removed.
        - Points behind either camera (positive z-coordinates) are removed.
        - Points occluded in the first camera's view (i.e., depth at projected pixel is smaller than the 
          actual camera-to-point distance) are removed.
    """
    _, _, W, H = intrinsics_0

    depth_0 = depth_0.flatten()

    mask = (
    (ps_0[:, 0] >= 0) & (ps_0[:, 0] < W) & # Within image width
    (ps_0[:, 1] >= 0) & (ps_0[:, 1] < H) & # Within image height
    (pcl_0_frame_cam1[:,2] < 0) & (pcl_1_frame_cam0[:,2] < 0) & # Infront of both cameras
    (depth_0 + depth_threshold >= torch.norm(pcl_1_frame_cam0, dim=1)) # Not occluded
    )

    return mask

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