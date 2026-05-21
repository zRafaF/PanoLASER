import torch
import numpy as np
import open3d as o3d

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=2, overlap=1):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.reset()
        
    def reset(self):
        """Clears the temporal memory for a new video sequence."""
        self.global_pcd = o3d.geometry.PointCloud()
        self.prev_raw_pts = None
        self.prev_global_pose = None
        self.is_first_window = True

    def _build_global_pcd(self, raw_pts, rgb, mask, global_pose):
        """Filters invalid pixels and transforms points into the global world space."""
        valid = mask.astype(bool)
        pts = raw_pts[valid]
        colors = rgb[valid] / 255.0
        
        # Apply global 4x4 pose matrix using homogeneous coordinates
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])
        pts_global = (global_pose @ pts_homo.T).T[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_global)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _get_global_pose(self, local_pose, R, t, scale):
        """Calculates the new global pose matrix given Kabsch outputs."""
        S = np.eye(4)
        S[:3, :3] *= scale
        
        T_align = np.eye(4)
        T_align[:3, :3] = R
        T_align[:3, 3] = t
        
        return T_align @ S @ local_pose

    def process_sequence(self, frames, masks):
        """Processes a full sequence using LASER Sliding Window architecture."""
        self.reset()
        
        for i in range(0, len(frames) - 1, self.window_size - self.overlap):
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"[Streaming Engine] Processing Window {i}...")
            preds = self.engine(window_frames)
            pts_list = preds["points"]
            poses = preds["poses"]
            
            if self.is_first_window:
                # First window anchors the global map
                for j in range(self.window_size):
                    pcd = self._build_global_pcd(pts_list[j], window_frames[j], window_masks[j], poses[j])
                    self.global_pcd += pcd
                    
                # Track the overlapping frame (Frame B)
                self.prev_raw_pts = pts_list[-1]
                self.prev_global_pose = poses[-1]
                self.is_first_window = False
                
            else:
                # Subsequent windows align to the overlapping frame (Frame B)
                curr_raw_pts = pts_list[0]
                curr_local_pose = poses[0]
                mask_torch = torch.from_numpy(window_masks[0])
                
                # 1. IRLS Scale Drift Correction (Comparing raw, unrotated geometry)
                scale_diff = align_cam_pts_irls(
                    torch.from_numpy(curr_raw_pts), 
                    torch.from_numpy(self.prev_raw_pts), 
                    mask_torch
                )
                
                # 2. Kabsch Rigid Alignment (Comparing camera anchors)
                R, t = register_camera_poses_kabsch(
                    np.expand_dims(curr_local_pose, 0), 
                    np.expand_dims(self.prev_global_pose, 0), 
                    scale=scale_diff
                )
                print(f"  -> Scale Drift Corrected: {scale_diff:.4f}")
                
                # 3. Stitch new frame (Frame C)
                new_raw_pts = pts_list[1]
                new_local_pose = poses[1]
                
                # Map Frame C's local pose to the Global Trajectory space
                new_global_pose = self._get_global_pose(new_local_pose, R, t, scale_diff)
                
                pcd = self._build_global_pcd(new_raw_pts, window_frames[1], window_masks[1], new_global_pose)
                self.global_pcd += pcd
                
                # Update tracking for the next iteration
                self.prev_raw_pts = new_raw_pts
                self.prev_global_pose = new_global_pose

        # Downsample final map
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.05)
        return self.global_pcd