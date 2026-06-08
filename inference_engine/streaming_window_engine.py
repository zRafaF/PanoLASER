import torch
import numpy as np
import open3d as o3d
import time

from .proximity_mapper import ProximityVoxelMap
from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .metricfication import estimate_metric_scale_from_floor
from .vram_profiler import VRAMProfiler

class StreamingWindowEngine:
    def __init__(self, vanilla_engine, window_size=16, overlap=2, device="cuda"):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        self.target_camera_height = 1.7 
        self.vram_tracker = VRAMProfiler()
        self.max_depth = 5.0 # Can be updated from UI
        
        print("[Engine] Initializing Proximity Voxel Engine...")
        self.reset()
        
    def reset(self):
        self.last_pcd = None
        
        # Initialize the new Proximity Mapper
        self.mapper = ProximityVoxelMap(voxel_size=0.04, max_depth=self.max_depth)
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.full_poses = []
        self.processed_indices = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0

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
            floor_scale, floor_conf = estimate_metric_scale_from_floor(
                pts_list[mid_idx], target_camera_height=self.target_camera_height
            )
            
            if self.is_first_window:
                if floor_scale is not None:
                    self.current_metric_scale = floor_scale
                    print(f"  > [Metric] Floor Detected | Conf: {floor_conf:.2f} | Scale: {floor_scale:.3f}")
                else:
                    first_depth = np.linalg.norm(pts_list[0], axis=-1)
                    valid_depths = first_depth[first_depth > 0.1]
                    if len(valid_depths) > 0:
                        self.current_metric_scale = 3.0 / np.median(valid_depths)
            else:
                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                relative_scale = self.current_metric_scale * (0.8 * 1.0 + 0.2 * clipped_scale)
                
                if floor_scale is not None and floor_conf > 0.4:
                    self.current_metric_scale = 0.9 * relative_scale + 0.1 * floor_scale
                else:
                    self.current_metric_scale = relative_scale

            metric_local_poses = []
            for p in poses:
                mp = p.copy()
                mp[:3, 3] *= self.current_metric_scale 
                metric_local_poses.append(mp)
                
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]

            anchor_pose = np.eye(4)
            if not self.is_first_window:
                src_cam_np = np.stack(canonical_poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=1.0)
                anchor_pose[:3, :3] = R_align
                anchor_pose[:3, 3] = t_align
                
            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            
            start_idx = 0 if self.is_first_window else self.overlap
            
            for j in range(self.window_size):
                global_pose = anchor_pose @ canonical_poses[j]
                
                if j >= start_idx:
                    self.trajectory.append(global_pose[:3, 3])
                    self.processed_indices.append(i + j)
                    self.full_poses.append(global_pose)
                    
                    # Flip Y and Z axes to match OpenCV -> World mapping
                    pose_aligned = global_pose.copy()
                    flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    pose_aligned[:3, :3] = pose_aligned[:3, :3] @ flip_R

                    scaled_pts = pts_list[j] * self.current_metric_scale
                    batch_depths.append(scaled_pts) 
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(pose_aligned) 
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            profiler["Scale_&_Pose_Math"] = time.time() - t2

            # --- Synchronous Proximity Integration ---
            t3 = time.time()
            self.mapper.integrate(batch_depths, batch_rgbs, batch_masks, batch_poses)
            profiler["Proximity_Mapping"] = time.time() - t3
            
            # --- Profiling & VRAM Output ---
            avg_alloc, max_alloc, avg_res, max_res = self.vram_tracker.stop()
            total_time = time.time() - t_win_start
            
            print("  --- Performance Profile ---")
            for k, v in profiler.items():
                print(f"    - {k:<30}: {v:.4f} sec")
            print(f"  >>> Total Cycle Time: {total_time:.4f} sec")
            print("  --- GPU Memory (VRAM) ---")
            print(f"    - Allocated : {avg_alloc:.2f} GB (Avg) | {max_alloc:.2f} GB (Peak)")
            print(f"    - Reserved  : {avg_res:.2f} GB (Avg) | {max_res:.2f} GB (Peak)")

            self.submap_count += 1
            
            # Live UI Update
            self.last_pcd = self.mapper.extract_point_cloud()

            yield_pcd = self.last_pcd
            if yield_pcd is None or len(yield_pcd.points) == 0:
                yield_pcd = o3d.geometry.PointCloud()
                yield_pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))

            # Yield an empty mesh during streaming to save UI cycle time
            yield o3d.geometry.TriangleMesh(), yield_pcd, np.array(self.trajectory), []

        print("[Engine] Final Sequence: Building High-Res Poisson Mesh...")
        final_mesh = self.mapper.extract_poisson_mesh(depth=10)
        final_pcd = self.mapper.extract_point_cloud()

        print(f"\n[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        yield final_mesh, final_pcd, np.array(self.trajectory), []