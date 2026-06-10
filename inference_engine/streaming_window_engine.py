import torch
import numpy as np
import open3d as o3d
import time

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .metricfication import estimate_metric_scale_from_floor
from .vram_profiler import VRAMProfiler
from .gaussian_mapper import PanoGaussianMapper 

class StreamingWindowEngine:
    def __init__(self, vanilla_engine, window_size=16, overlap=2, device="cuda"):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        self.target_camera_height = 1.7 
        self.vram_tracker = VRAMProfiler()
        
        # User Mutable Tuning Parameters
        self.max_depth = 4.5
        self.min_translation_m = 0.10
        
        # NEW: Decimation factor for 3DGS seeding. 
        # 0.05 means we only keep 5% of the dense VGGT points.
        self.seed_keep_ratio = 0.05 
        
        print("[Engine] Initializing 3D Gaussian Splatting Streaming Engine...")
        self.reset()
        
    def reset(self):
        self.gs_mapper = PanoGaussianMapper(device=self.device, face_size=512)
        self.last_pcd = None
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.full_poses = []
        self.processed_indices = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        self.last_integrated_position = None

    def process_sequence(self, frames, masks):
        self.reset()
        num_frames = len(frames)
        t_seq_start = time.time()
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            self.vram_tracker.start()
            t_win_start = time.time()
            profiler = {}

            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n==========================================")
            print(f"[Engine] Processing Submap {self.submap_count}...")
            
            t1 = time.time()
            with torch.inference_mode():
                preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            profiler["VGGT_Inference"] = time.time() - t1

            pts_list = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["points"]]
            poses = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["poses"]]
            del preds 
            
            t2 = time.time()
            mid_idx = self.window_size // 2
            floor_scale, floor_conf = estimate_metric_scale_from_floor(pts_list[mid_idx], target_camera_height=self.target_camera_height)
            
            if self.is_first_window:
                if floor_scale is not None:
                    self.current_metric_scale = floor_scale
                else:
                    first_depth = np.linalg.norm(pts_list[0], axis=-1)
                    valid_depths = first_depth[first_depth > 0.1]
                    if len(valid_depths) > 0: self.current_metric_scale = 3.0 / np.median(valid_depths)
            else:
                raw_scale_diff = align_cam_pts_irls(torch.from_numpy(pts_list[0].copy()), torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), torch.from_numpy(window_masks[0].copy()))
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                relative_scale = self.current_metric_scale * (0.8 * 1.0 + 0.2 * clipped_scale)
                self.current_metric_scale = 0.9 * relative_scale + 0.1 * floor_scale if (floor_scale is not None and floor_conf > 0.4) else relative_scale

            metric_local_poses = [p.copy() for p in poses]
            for p in metric_local_poses: p[:3, 3] *= self.current_metric_scale 
                
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]

            anchor_pose = np.eye(4)
            if not self.is_first_window:
                R_align, t_align = register_camera_poses_kabsch(np.stack(canonical_poses[:self.overlap]), np.stack(self.prev_overlap_global_poses), scale=1.0)
                anchor_pose[:3, :3] = R_align
                anchor_pose[:3, 3] = t_align
                
            batch_rgbs, batch_poses = [], []
            seed_pts, seed_colors = [], []
            start_idx = 0 if self.is_first_window else self.overlap
            
            for j in range(self.window_size):
                global_pose = anchor_pose @ canonical_poses[j]
                
                if j >= start_idx:
                    current_pos = global_pose[:3, 3]
                    self.trajectory.append(current_pos)
                    self.processed_indices.append(i + j)
                    self.full_poses.append(global_pose)
                    
                    dist = 999.0 if self.last_integrated_position is None else np.linalg.norm(current_pos - self.last_integrated_position)
                            
                    if dist >= self.min_translation_m:
                        self.last_integrated_position = current_pos.copy()
                        
                        gs_pose = global_pose.copy()
                        if gs_pose[1, 1] < 0:
                            flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                            gs_pose[:3, :3] = gs_pose[:3, :3] @ flip_R

                        # ======================================================
                        # POINT PRUNING AND DECIMATION
                        # ======================================================
                        scaled_pts = pts_list[j] * self.current_metric_scale
                        rgb_frame = window_frames[j]
                        mask = window_masks[j]
                        
                        point_depths = np.linalg.norm(scaled_pts, axis=-1)
                        
                        # 1. Enforce Max Depth and Minimum Valid Range
                        valid_mask = (mask > 0) & (point_depths > 0.2) & (point_depths <= self.max_depth)
                        
                        valid_pts = scaled_pts[valid_mask]
                        valid_colors = rgb_frame[valid_mask]
                        
                        num_valid = len(valid_pts)
                        if num_valid > 0:
                            # 2. Random Decimation to prevent VRAM OOM
                            num_to_keep = max(1, int(num_valid * self.seed_keep_ratio))
                            keep_indices = np.random.choice(num_valid, size=num_to_keep, replace=False)
                            
                            pruned_pts = valid_pts[keep_indices]
                            pruned_colors = valid_colors[keep_indices]
                            
                            # Transform to global space
                            global_pts = (pruned_pts @ gs_pose[:3, :3].T) + gs_pose[:3, 3]
                            seed_pts.append(global_pts)
                            
                            normalized_colors = np.clip(pruned_colors / 255.0, 1e-4, 1.0 - 1e-4)
                            seed_colors.append(np.log(normalized_colors / (1 - normalized_colors)))

                        batch_rgbs.append(rgb_frame)
                        batch_poses.append(gs_pose) 
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap: self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            profiler["Scale_&_Pose_Math"] = time.time() - t2

            t3 = time.time()
            if len(batch_poses) > 0 and len(seed_pts) > 0:
                print(f"  > [3DGS] Initializing {sum([len(p) for p in seed_pts])} new Gaussians...")
                self.gs_mapper.seed_new_points(np.concatenate(seed_pts), np.concatenate(seed_colors))
                
                print(f"  > [3DGS] Training Submap on {len(batch_poses)} Keyframes (15 Steps)...")
                self.gs_mapper.train_submap(batch_rgbs, batch_poses, iterations=15)
            profiler["Gaussian_Splat_Training"] = time.time() - t3
            
            avg_alloc, max_alloc, avg_res, max_res = self.vram_tracker.stop()
            total_time = time.time() - t_win_start
            
            print("  --- Performance Profile ---")
            for k, v in profiler.items(): print(f"    - {k:<30}: {v:.4f} sec")
            print(f"  >>> Total Cycle Time: {total_time:.4f} sec")
            print("  --- GPU Memory (VRAM) ---")
            print(f"    - Allocated : {avg_alloc:.2f} GB (Avg) | {max_alloc:.2f} GB (Peak)")
            print(f"    - Reserved  : {avg_res:.2f} GB (Avg) | {max_res:.2f} GB (Peak)")
            
            self.submap_count += 1
            
            self.last_pcd = self.gs_mapper.get_o3d_pointcloud()

            yield o3d.geometry.TriangleMesh(), self.last_pcd, np.array(self.trajectory), []

        print(f"\n[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        yield o3d.geometry.TriangleMesh(), self.last_pcd, np.array(self.trajectory), []