import torch
import numpy as np
import open3d as o3d

from .inference_utils import align_cam_pts_irls

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=2, overlap=1):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        
        if overlap != 1:
            raise ValueError("This engine is currently optimized for exact 1-frame odometry chaining.")
        self.reset()
        
    def reset(self):
        """Clears the temporal memory for a new video sequence."""
        self.global_pcd = o3d.geometry.PointCloud()
        self.prev_raw_pts = None
        self.prev_global_pose = None
        self.is_first_window = True

    def _build_global_pcd(self, raw_pts, rgb, mask, global_pose, scale=1.0):
        """Filters invalid pixels, applies scale, and transforms into global space."""
        valid = mask.astype(bool)
        pts = raw_pts[valid] * scale
        colors = rgb[valid] / 255.0
        
        # Apply global 4x4 pose matrix using homogeneous coordinates
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])
        pts_global = (global_pose @ pts_homo.T).T[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_global)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def process_sequence(self, frames, masks):
        """Processes a sequence using Exact Odometry Chaining & IRLS Scale Alignment."""
        self.reset()
        
        for i in range(0, len(frames) - 1, self.window_size - self.overlap):
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\\n[Streaming Engine] Processing Window {i}...")
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
                # 1. IRLS Scale Drift Correction
                curr_raw_pts = pts_list[0]
                mask_torch = torch.from_numpy(window_masks[0])
                
                scale_diff = align_cam_pts_irls(
                    torch.from_numpy(curr_raw_pts), 
                    torch.from_numpy(self.prev_raw_pts), 
                    mask_torch
                )
                print(f"  -> Scale Drift Corrected: {scale_diff:.4f}")
                
                # 2. Exact Odometry Chaining for the new frame (Frame C)
                new_raw_pts = pts_list[1]
                new_local_pose = poses[1]
                
                # Scale the Translation component of the local pose
                local_pose_scaled = new_local_pose.copy()
                local_pose_scaled[:3, 3] *= scale_diff
                
                # Chain the scaled local transform onto the global trajectory anchor
                new_global_pose = self.prev_global_pose @ local_pose_scaled
                
                # Build the scaled, globally aligned point cloud
                pcd = self._build_global_pcd(new_raw_pts, window_frames[1], window_masks[1], new_global_pose, scale=scale_diff)
                self.global_pcd += pcd
                
                # Update tracking for the next iteration
                self.prev_raw_pts = new_raw_pts
                self.prev_global_pose = new_global_pose

        # Downsample final map
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.05)
        print("[Streaming Engine] Sequence processing complete.")
        return self.global_pcd