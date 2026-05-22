import torch
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .tsdf_volume import FastStaticTSDF  

class PanoStreamingEngine:
    # [UPGRADE 1]: Window size bumped to 16 for massive batch processing speeds
    def __init__(self, vanilla_engine, window_size=16, overlap=2):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # [UPGRADE 2]: Dedicated background thread for TSDF integration
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        
        if self.overlap < 2:
            print("WARNING: Kabsch requires overlap >= 2")
        self.reset()
        
    def reset(self):
        # Wait for any lingering background mapping to finish before resetting
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            
        self.tsdf = FastStaticTSDF(voxel_size=0.02, margin=0.08, max_depth=6.0, device=self.device)
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.is_first_window = True
        self.metric_scale = 1.0
        
        self.last_integrated_pose = None

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
        if self.last_integrated_pose is None:
            return True
        dt = np.linalg.norm(current_pose[:3, 3] - self.last_integrated_pose[:3, 3])
        R_diff = self.last_integrated_pose[:3, :3].T @ current_pose[:3, :3]
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle = np.degrees(np.arccos((trace - 1.0) / 2.0))
        return dt > trans_thresh or angle > rot_thresh

    def _async_tsdf_task(self, depth_maps, rgb_frames, masks, poses):
        """This function runs entirely in the background so the main thread can keep inference going."""
        t_start = time.time()
        integrated = 0
        for j in range(len(poses)):
            if self._is_keyframe(poses[j]):
                self.tsdf.integrate(depth_maps[j], rgb_frames[j], masks[j], poses[j])
                self.last_integrated_pose = poses[j]
                integrated += 1
                
        torch.cuda.synchronize()
        return integrated, time.time() - t_start

    def process_sequence(self, frames, masks):
        self.reset()
        num_frames = len(frames)
        t_seq_start = time.time()
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            t_win_start = time.time()
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n[Streaming Engine] Processing Submap {i//(self.window_size-self.overlap)} (Frames {i} to {i+self.window_size})...")
            
            # --- 1. GPU INFERENCE (Main Thread) ---
            t_gpu_start = time.time()
            preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            print(f"  [Profile] VGGT Submap Inference ({self.window_size} frames): {time.time() - t_gpu_start:.4f} sec")
            
            pts_list = preds["points"] 
            poses = preds["poses"]
            
            # Buffers to send to the TSDF Thread
            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            
            if self.is_first_window:
                self.prev_overlap_global_poses = []
                self.world_anchor = np.linalg.inv(poses[0])
                
                first_depth = np.linalg.norm(pts_list[0], axis=-1)
                valid_depths = first_depth[first_depth > 0.1]
                if len(valid_depths) > 0:
                    self.metric_scale = 3.0 / np.median(valid_depths)
                
                for j in range(self.window_size):
                    aligned_pose = self.world_anchor @ poses[j]
                    scaled_pts = pts_list[j] * self.metric_scale
                    
                    batch_depths.append(np.linalg.norm(scaled_pts, axis=-1))
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(aligned_pose)
                    self.prev_overlap_global_poses.append(aligned_pose)
                    
                self.prev_overlap_raw_pts = [pts * self.metric_scale for pts in pts_list[-self.overlap:]]
                self.prev_overlap_global_poses = self.prev_overlap_global_poses[-self.overlap:]
                self.is_first_window = False
                
            else:
                scaled_incoming_pts = [pts * self.metric_scale for pts in pts_list]
                
                # We must wait for the previous TSDF thread to finish so scale doesn't drift uncontrollably
                if self.tsdf_future is not None:
                    integ_count, t_time = self.tsdf_future.result()
                    print(f"  [Profile] Background TSDF Thread mapped {integ_count} keyframes in {t_time:.4f} sec.")
                
                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(scaled_incoming_pts[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                scale_diff = 0.8 * 1.0 + 0.2 * clipped_scale
                
                src_cam_np = np.stack(poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=scale_diff)
                
                new_global_poses = []
                for j in range(self.window_size):
                    global_pose = self._apply_sim3_to_pose(poses[j], R_align, t_align, scale_diff)
                    new_global_poses.append(global_pose)
                    
                    if j >= self.overlap:
                        batch_depths.append(np.linalg.norm(scaled_incoming_pts[j], axis=-1))
                        batch_rgbs.append(window_frames[j])
                        batch_masks.append(window_masks[j])
                        batch_poses.append(global_pose)
                        
                self.prev_overlap_raw_pts = scaled_incoming_pts[-self.overlap:]
                self.prev_overlap_global_poses = new_global_poses[-self.overlap:]

            # --- 2. ASYNC DISPATCH ---
            # Fire and forget the integration into the background thread. 
            # The main thread immediately loops back to run VGGT on the next submap!
            self.tsdf_future = self.tsdf_executor.submit(
                self._async_tsdf_task, batch_depths, batch_rgbs, batch_masks, batch_poses
            )
            
            print(f"  [Profile] ----- Engine Cycle Time: {time.time() - t_win_start:.4f} sec -----")

        # Catch the very last background mapping task before returning
        if self.tsdf_future is not None:
            self.tsdf_future.result()

        print(f"[Streaming Engine] Sequence completely mapped in {time.time() - t_seq_start:.4f} sec.")
        return self.tsdf.extract_point_cloud(surface_threshold=0.02)