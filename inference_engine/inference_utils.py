import torch
import numpy as np

# Correct relative paths pointing down into the utils folder
from .utils.geometry import register_camera_poses_kabsch, apply_sim3_to_pose
from .utils.rsa import refine_depth_segments

def align_cam_pts_irls(
        src_pts: torch.Tensor,  
        tgt_pts: torch.Tensor,  
        mask: torch.Tensor = None,  
        iters: int = 10,
        eps: float = 1e-8,
        stop_tol: float = 0.05,
        clamp_min: float = 1e-6
) -> float:
    """
    Calculates the global scale difference between two overlapping 3D point clouds 
    using Iteratively Reweighted Least Squares (IRLS).
    """
    if mask is not None:
        if src_pts.dim() == 4:
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(src_pts)
            src_valid = src_pts[mask_expanded.bool()].view(-1, 3)
            tgt_valid = tgt_pts[mask_expanded.bool()].view(-1, 3)
        else:
            src_valid = src_pts[mask.bool()]
            tgt_valid = tgt_pts[mask.bool()]
    else:
        src_valid = src_pts.reshape(-1, 3)
        tgt_valid = tgt_pts.reshape(-1, 3)

    src_np = src_valid.cpu().numpy()
    tgt_np = tgt_valid.cpu().numpy()
    
    num = np.nanmean(np.linalg.norm(tgt_np, axis=-1))
    den = np.nanmean(np.linalg.norm(src_np, axis=-1))
    s_d = np.maximum(num / den, clamp_min)

    for _ in range(iters):
        d_res = s_d * src_np - tgt_np
        res = np.linalg.norm(d_res, axis=-1) + eps
        w = 1.0 / res
        
        num = (w * np.linalg.norm(src_np, axis=-1) * np.linalg.norm(tgt_np, axis=-1)).sum()
        den = (w * np.linalg.norm(src_np, axis=-1) ** 2).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        
        if abs(s_d_new - s_d) < stop_tol:
            break
        s_d = s_d_new
        
    return float(s_d)

def register_extrinsic_windows(
        src_pcd_overlap: torch.Tensor,
        tgt_pcd_overlap: torch.Tensor,
        src_cam_overlap: torch.Tensor,
        tgt_cam_overlap: torch.Tensor,
        mask: torch.Tensor,
):
    s_d = align_cam_pts_irls(tgt_pcd_overlap, src_pcd_overlap, mask)

    src_cam_np = src_cam_overlap.cpu().numpy()
    tgt_cam_np = tgt_cam_overlap.cpu().numpy()
    
    R, t = register_camera_poses_kabsch(tgt_cam_np, src_cam_np, scale=s_d)
    
    R_torch = torch.from_numpy(R).float().to(src_pcd_overlap.device)
    t_torch = torch.from_numpy(t).float().to(src_pcd_overlap.device)
    
    return s_d, R_torch, t_torch