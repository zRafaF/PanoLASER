import torch
import numpy as np

# --- PanoLASER Spherical Math ---
def unproject_equirectangular_to_points(depth_map: np.ndarray) -> np.ndarray:
    """
    Converts an equirectangular radial depth map into 3D Cartesian coordinates.
    Assumes OpenCV coordinate system.
    """
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert pixels to spherical angles
    theta = (u / W - 0.5) * 2 * np.pi
    phi = (v / H - 0.5) * np.pi
    
    # Spherical to Cartesian
    X = depth_map * np.cos(phi) * np.sin(theta)
    Y = depth_map * np.sin(phi)
    Z = depth_map * np.cos(phi) * np.cos(theta)
    
    return np.stack([X, Y, Z], axis=-1)


# --- Original LASER Alignment Math ---
def homogenize_points(points):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def homogenize_points_np(points):
    """Convert batched points (xyz) to (xyz1)."""
    return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)

def register_camera_poses_kabsch(src_cam_poses: np.ndarray, tgt_cam_poses: np.ndarray, scale=1.0):
    """
    Aligns two sets of camera poses using Kabsch algorithm.
    Enhanced with full 3D coordinate frame locking to prevent collinear degeneracy.
    """
    assert src_cam_poses.shape == tgt_cam_poses.shape
    
    # 1. Extract and scale translation
    src_pos = src_cam_poses[:, :3, 3] * scale
    tgt_pos = tgt_cam_poses[:, :3, 3]

    # 2. Extract full coordinate frame (X, Y, Z axes) for orientation locking
    # This prevents the submap from spinning if the camera trajectory is a straight line.
    src_x = src_cam_poses[:, :3, :3] @ np.array([1., 0., 0.])
    src_y = src_cam_poses[:, :3, :3] @ np.array([0., 1., 0.])
    src_z = src_cam_poses[:, :3, :3] @ np.array([0., 0., 1.])

    tgt_x = tgt_cam_poses[:, :3, :3] @ np.array([1., 0., 0.])
    tgt_y = tgt_cam_poses[:, :3, :3] @ np.array([0., 1., 0.])
    tgt_z = tgt_cam_poses[:, :3, :3] @ np.array([0., 0., 1.])

    # 3. Build Point Clouds (Position + Triad)
    # We append unit vectors to the scaled positions.
    src_pts = np.concatenate([src_pos, src_pos + src_x, src_pos + src_y, src_pos + src_z], axis=0)
    tgt_pts = np.concatenate([tgt_pos, tgt_pos + tgt_x, tgt_pos + tgt_y, tgt_pos + tgt_z], axis=0)

    # 4. Standard Kabsch SVD
    src_centroid = np.mean(src_pts, axis=0)
    tgt_centroid = np.mean(tgt_pts, axis=0)

    src_pts_centered = src_pts - src_centroid
    tgt_pts_centered = tgt_pts - tgt_centroid

    H = src_pts_centered.T @ tgt_pts_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid
    
    return R, t
def apply_scale_with_so3(poses, R, scale):
    """Apply scale to camera poses in a rotated basis."""
    device = poses.device
    S = torch.eye(4, device=device)
    S[:3, :3] = scale * torch.eye(3, device=device)

    R_h = torch.eye(4, device=device)
    R_h[:3, :3] = R
    S_rot = R_h.T @ S @ R