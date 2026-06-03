import torch
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from .nvblox_tsdf import NvbloxPanoTSDF
from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch

class StreamingWindowEngine:
    def __init__(self, vanilla_engine, window_size=16, overlap=2, device="cuda"):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        
        print("[Engine] Initializing Pure Streaming Odometry Engine (No Loop Closure)...")
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            
        self.tsdf = NvbloxPanoTSDF(voxel_size_m=0.015, max_depth=5.0, device=self.device)
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        self.last_integrated_pose = None

    def _is_keyframe(self, current_pose, trans_thresh=0.15, rot_thresh=5.0):
        if self.last_integrated_pose is None:
            return True
        dt = np.linalg.norm(current_pose[:3, 3] - self.last_integrated_pose[:3, 3])
        R_diff = self.last_integrated_pose[:3, :3].T @ current_pose[:3, :3]
        trace = np.clip(np.trace(R_diff), -1.0, 3.0)
        angle = np.degrees(np.arccos((trace - 1.0) / 2.0))
        return dt > trans_thresh or angle > rot_thresh

    def _async_tsdf_task(self, depth_maps, rgb_frames, masks, poses):
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
            
            print(f"\n[Engine] Processing Submap {self.submap_count}...")
            
            # Flush VRAM before network inference
            torch.cuda.empty_cache() 
            
            t_gpu = time.time()
            with torch.inference_mode():
                preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            print(f"  [Profile] VGGT Inference: {time.time() - t_gpu:.4f} sec")

            # Extract data and immediately delete graph tensors to free memory
            pts_list = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["points"]]
            poses = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["poses"]]
            del preds 
            torch.cuda.empty_cache()
            
            # --- SCALE OPTIMIZATION ---
            if self.is_first_window:
                first_depth = np.linalg.norm(pts_list[0], axis=-1)
                valid_depths = first_depth[first_depth > 0.1]
                if len(valid_depths) > 0:
                    self.current_metric_scale = 3.0 / np.median(valid_depths)
            else:
                if self.tsdf_future is not None:
                    self.tsdf_future.result()
                    
                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                self.current_metric_scale *= (0.8 * 1.0 + 0.2 * clipped_scale)

            # --- POSE CALCULATION ---
            metric_local_poses = []
            for p in poses:
                mp = p.copy()
                mp[:3, 3] *= self.current_metric_scale 
                metric_local_poses.append(mp)
                
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]

            # Stitch window to global trajectory via overlapping frames
            anchor_pose = np.eye(4)
            if not self.is_first_window:
                src_cam_np = np.stack(canonical_poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=1.0)
                anchor_pose[:3, :3] = R_align
                anchor_pose[:3, 3] = t_align
                
            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            
            for j in range(self.window_size):
                global_pose = anchor_pose @ canonical_poses[j]
                
                if j >= (0 if self.is_first_window else self.overlap):
                    self.trajectory.append(global_pose[:3, 3])
                
                tsdf_pose = global_pose.copy()
                if tsdf_pose[1, 1] < 0:
                    flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    tsdf_pose[:3, :3] = tsdf_pose[:3, :3] @ flip_R

                scaled_pts = pts_list[j] * self.current_metric_scale
                depth_map = np.linalg.norm(scaled_pts, axis=-1)
                
                if j >= (0 if self.is_first_window else self.overlap):
                    batch_depths.append(depth_map)
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(tsdf_pose) 
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False

            # --- ASYNC INTEGRATION ---
            self.tsdf_future = self.tsdf_executor.submit(
                self._async_tsdf_task, batch_depths, batch_rgbs, batch_masks, batch_poses
            )
            
            self.submap_count += 1
            print(f"  [Profile] Cycle Time: {time.time() - t_win_start:.4f} sec")

            if self.tsdf_future is not None:
                self.tsdf_future.result()

            torch.cuda.empty_cache()

            # Empty list returned for lc_edges since loop closure is disabled
            live_pcd = self.tsdf.extract_point_cloud(viz_voxel_scale=4.0)
            yield None, live_pcd, np.array(self.trajectory), []

        print(f"[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        final_mesh = self.tsdf.extract_mesh() 
        yield final_mesh, None, np.array(self.trajectory), []