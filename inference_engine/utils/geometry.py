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
    assert src_cam_poses.shape == tgt_cam_poses.shape
    
    src_cam_pos = src_cam_poses[:, :3, 3]
    src_cam_view = src_cam_poses[:, :3, :3] @ np.array([0., 0., -1.])
    src_cam_view_norm = src_cam_view / np.linalg.norm(src_cam_view, axis=-1, keepdims=True)

    tgt_cam_pos = tgt_cam_poses[:, :3, 3]
    tgt_cam_view = tgt_cam_poses[:, :3, :3] @ np.array([0., 0., -1.])
    tgt_cam_view_norm = tgt_cam_view / np.linalg.norm(tgt_cam_view, axis=-1, keepdims=True)

    src_pts = np.concatenate([src_cam_pos, src_cam_pos + src_cam_view_norm], axis=0) * scale
    tgt_pts = np.concatenate([tgt_cam_pos, tgt_cam_pos + tgt_cam_view_norm], axis=0)

    src_centroid = np.mean(src_pts, axis=0)
    tgt_centroid = np.mean(tgt_pts, axis=0)

    src_pts_centered = src_pts - src_centroid
    tgt_pts_centered = tgt_pts - tgt_centroid

    H = src_pts_centered.T @ tgt_pts_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
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