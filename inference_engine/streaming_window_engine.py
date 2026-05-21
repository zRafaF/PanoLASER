import torch
import numpy as np
import open3d as o3d

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=3, overlap=2):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        
        if self.overlap < 2:
            print("WARNING: Kabsch requires an overlap of >= 2 to guarantee a stable 3D rotation.")
        self.reset()
        
    def reset(self):
        """Clears the temporal memory for a new video sequence."""
        self.global_pcd = o3d.geometry.PointCloud()
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.prev_overlap_masks = []
        self.is_first_window = True

    def _apply_sim3_to_pose(self, local_pose, R_align, t_align, scale):
        """
        Applies Sim(3) with an Uprightness Constraint to prevent floor/ceiling flipping.
        """
        # 1. Scale ONLY the translation component
        L_scaled = local_pose.copy()
        L_scaled[:3, 3] *= scale

        # 2. Construct Rigid Alignment
        T_align = np.eye(4)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = t_align

        # 3. Calculate candidate pose
        candidate_pose = T_align @ L_scaled
        
        # 4. UP-VECTOR CONSISTENCY CHECK
        # The 'Up' vector in world space is (0, 1, 0). 
        # In the camera pose matrix, this is stored in the 2nd column (index 1) of the rotation part.
        up_vector_world = candidate_pose[:3, 1] 
        
        # If the Y-component is negative, the camera thinks 'up' is 'down'
        if up_vector_world[1] < 0:
            print("  -> [Geometry Warning] Detected upside-down frame, applying 180° flip.")
            # Apply a 180-degree rotation around the X-axis to flip the frame
            flip_R = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            candidate_pose[:3, :3] = candidate_pose[:3, :3] @ flip_R
            
        return candidate_pose

    def _build_global_pcd(self, raw_pts, rgb, mask, global_pose, scale=1.0):
        """Filters invalid pixels, applies scale, and transforms into global space."""
        valid = mask.astype(bool)
        # 1. Scale the local geometry points
        pts = raw_pts[valid] * scale
        colors = rgb[valid] / 255.0
        
        # 2. Transform to global space using the strict SE(3) pose
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])
        pts_global = (global_pose @ pts_homo.T).T[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_global)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def process_sequence(self, frames, masks):
        """Processes a sequence using exact Kabsch alignment and Sim(3) Odometry."""
        self.reset()
        step = self.window_size - self.overlap
        
        for i in range(0, len(frames) - self.window_size + 1, step):
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n[Streaming Engine] Processing Window (Frames {i} to {i + self.window_size - 1})...")
            preds = self.engine(window_frames)
            pts_list = preds["points"]
            poses = preds["poses"]
            
            if self.is_first_window:
                # First window anchors the global map
                for j in range(self.window_size):
                    pcd = self._build_global_pcd(pts_list[j], window_frames[j], window_masks[j], poses[j])
                    self.global_pcd += pcd
                    
                # Store the overlapping frames for the next window
                self.prev_overlap_raw_pts = pts_list[-self.overlap:]
                self.prev_overlap_global_poses = list(poses[-self.overlap:])
                self.prev_overlap_masks = window_masks[-self.overlap:]
                self.is_first_window = False
                
            else:
                curr_overlap_raw_pts = pts_list[:self.overlap]
                curr_overlap_local_poses = poses[:self.overlap]
                curr_overlap_masks = window_masks[:self.overlap]
                
                # 1. IRLS Scale Drift Correction
                scale_diffs = []
                for k in range(self.overlap):
                    s = align_cam_pts_irls(
                        torch.from_numpy(curr_overlap_raw_pts[k]), 
                        torch.from_numpy(self.prev_overlap_raw_pts[k]), 
                        torch.from_numpy(curr_overlap_masks[k])
                    )
                    scale_diffs.append(s)
                scale_diff = float(np.median(scale_diffs))
                
                # 2. Kabsch Rigid Alignment on multiple camera poses
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                src_cam_np = np.stack(curr_overlap_local_poses)
                R, t = register_camera_poses_kabsch(tgt_cam_np, src_cam_np, scale=scale_diff)
                print(f"  -> Kabsch Rotation Lock & Scale Corrected: {scale_diff:.4f}")
                
                # 3. Stitch new frames
                new_global_poses = []
                for j in range(self.overlap, self.window_size):
                    new_raw_pts = pts_list[j]
                    new_local_pose = poses[j]
                    
                    # Apply pure Sim(3) transform to the camera pose
                    new_global_pose = self._apply_sim3_to_pose(new_local_pose, R, t, scale_diff)
                    new_global_poses.append(new_global_pose)
                    
                    pcd = self._build_global_pcd(new_raw_pts, window_frames[j], window_masks[j], new_global_pose, scale=scale_diff)
                    self.global_pcd += pcd
                
                # 4. Update tracking state for next iteration
                self.prev_overlap_raw_pts = pts_list[-self.overlap:]
                window_global_poses = self.prev_overlap_global_poses + new_global_poses
                self.prev_overlap_global_poses = window_global_poses[-self.overlap:]
                self.prev_overlap_masks = window_masks[-self.overlap:]

        # Downsample final map
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.05)
        print("[Streaming Engine] Sequence processing complete.")
        return self.global_pcd