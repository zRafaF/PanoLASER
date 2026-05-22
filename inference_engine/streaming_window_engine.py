import torch
import numpy as np
import open3d as o3d
import time

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .tsdf_volume import FastStaticTSDF  

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=3, overlap=2):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.overlap < 2:
            print("WARNING: Kabsch requires overlap >= 2")
        self.reset()
        
    def reset(self):
        self.tsdf = FastStaticTSDF(voxel_size=0.02, margin=0.08, max_depth=6.0, device=self.device)
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.is_first_window = True
        self.metric_scale = 1.0
        
        # --- SLAM STATE TRACKING ---
        self.last_integrated_pose = None
        self.running_scale_diff = 1.0  # Momentum tracker for scale

    def _apply_sim3_to_pose(self, local_pose, R_align, t_align, scale):
        L_scaled = local_pose.copy()
        L_scaled[:3, 3] *= scale
        T_align = np.eye(4)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = t_align
        candidate_pose = T_align @ L_scaled
        up_vector_world = candidate_pose[:3, 1] 
        if up_vector_world[1] < 0:
            flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            candidate_pose[:3, :3] = candidate_pose[:3, :3] @ flip_R
        return candidate_pose

    def _is_keyframe(self, current_pose, trans_thresh=0.15, rot_thresh=5.0):
        """SLAM Heuristic: Only integrate if the camera has moved significantly."""
        if self.last_integrated_pose is None:
            return True
            
        # 1. Translation Distance (meters)
        dt = np.linalg.norm(current_pose[:3, 3] - self.last_integrated_pose[:3, 3])
        
        # 2. Rotation Difference (degrees)
        R_diff = self.last_integrated_pose[:3, :3].T @ current_pose[:3, :3]
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle = np.degrees(np.arccos((trace - 1.0) / 2.0))
        
        return dt > trans_thresh or angle > rot_thresh

    def process_sequence(self, frames, masks):
        self.reset()
        num_frames = len(frames)
        t_seq_start = time.time()
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            t_win_start = time.time()
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n[Streaming Engine] Processing Window {i}...")
            
            t_gpu_start = time.time()
            preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            
            pts_list = preds["points"] 
            poses = preds["poses"]
            
            if self.is_first_window:
                self.prev_overlap_global_poses = []
                self.world_anchor = np.linalg.inv(poses[0])
                
                first_depth = np.linalg.norm(pts_list[0], axis=-1)
                valid_depths = first_depth[first_depth > 0.1]
                
                if len(valid_depths) > 0:
                    self.metric_scale = 3.0 / np.median(valid_depths)
                print(f"  -> [Grounding] Auto-calibrated metric scale factor: {self.metric_scale:.4f}")
                
                for j in range(self.window_size):
                    aligned_pose = self.world_anchor @ poses[j]
                    scaled_pts = pts_list[j] * self.metric_scale
                    depth_map = np.linalg.norm(scaled_pts, axis=-1)
                    
                    # Keyframing Check
                    if self._is_keyframe(aligned_pose):
                        self.tsdf.integrate(depth_map, window_frames[j], window_masks[j], aligned_pose)
                        self.last_integrated_pose = aligned_pose
                        print(f"  -> [SLAM] Frame {i+j} integrated as Keyframe.")
                    
                    self.prev_overlap_global_poses.append(aligned_pose)
                    
                self.prev_overlap_raw_pts = [pts * self.metric_scale for pts in pts_list[-self.overlap:]]
                self.prev_overlap_global_poses = self.prev_overlap_global_poses[-self.overlap:]
                self.is_first_window = False
                
            else:
                scaled_incoming_pts = [pts * self.metric_scale for pts in pts_list]
                
                # --- SLAM HEURISTIC: ROBUST SCALE INERTIA ---
                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(scaled_incoming_pts[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                
                # 1. Hard clip: Never allow scale to jump more than 5% per window
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                # 2. Momentum: Pull the scale heavily towards 1.0 (assuming constant velocity)
                scale_diff = 0.8 * 1.0 + 0.2 * clipped_scale
                
                src_cam_np = np.stack(poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=scale_diff)
                
                new_global_poses = []
                for j in range(self.window_size):
                    global_pose = self._apply_sim3_to_pose(poses[j], R_align, t_align, scale_diff)
                    new_global_poses.append(global_pose)
                    
                    if j >= self.overlap:
                        depth_map = np.linalg.norm(scaled_incoming_pts[j], axis=-1)
                        
                        # Keyframing Check
                        if self._is_keyframe(global_pose):
                            self.tsdf.integrate(depth_map, window_frames[j], window_masks[j], global_pose)
                            self.last_integrated_pose = global_pose
                            print(f"  -> [SLAM] Frame {i+j} integrated as Keyframe.")
                        else:
                            print(f"  -> [SLAM] Frame {i+j} skipped (Insufficient movement).")
                        
                self.prev_overlap_raw_pts = scaled_incoming_pts[-self.overlap:]
                self.prev_overlap_global_poses = new_global_poses[-self.overlap:]

        print(f"[Streaming Engine] Sequence processing complete in {time.time() - t_seq_start:.4f} sec.")
        return self.tsdf.extract_point_cloud(surface_threshold=0.02)