import torch
import numpy as np
import open3d as o3d
import time

from .tsdf_volume import FastStaticTSDF
from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch

# Import our new metric logic
from .metricfication import estimate_metric_scale_from_floor

class StreamingWindowEngine:
    def __init__(self, vanilla_engine, window_size=16, overlap=2, device="cuda"):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        # You can eventually make this a dynamic input variable from the UI
        self.target_camera_height = 1.7 
        
        print("[Engine] Initializing PyTorch Engine with Dynamic Metricfication...")
        self.reset()
        
    def reset(self):
        self.tsdf = FastStaticTSDF(voxel_size=0.02, max_depth=5.0, device=self.device)
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        
        torch.cuda.empty_cache()

    def process_sequence(self, frames, masks):
        self.reset()
        num_frames = len(frames)
        t_seq_start = time.time()
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            t_win_start = time.time()
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n[Engine] Processing Submap {self.submap_count}...")
            torch.cuda.empty_cache() 
            
            t_gpu = time.time()
            with torch.inference_mode():
                preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            print(f"  [Profile] VGGT Inference: {time.time() - t_gpu:.4f} sec")

            pts_list = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["points"]]
            poses = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["poses"]]
            del preds 
            torch.cuda.empty_cache()
            
            # =========================================================
            # METRICFICATION & DYNAMIC SCALE CORRECTION
            # =========================================================
            mid_idx = self.window_size // 2
            floor_scale, floor_conf = estimate_metric_scale_from_floor(
                pts_list[mid_idx], 
                target_camera_height=self.target_camera_height
            )
            
            if self.is_first_window:
                if floor_scale is not None:
                    self.current_metric_scale = floor_scale
                    print(f"  [Metric] INITIALIZED via Floor | Conf: {floor_conf:.2f} | Scale: {floor_scale:.3f}")
                else:
                    print("  [Metric] WARNING: No clear floor on Frame 1. Using fallback median.")
                    first_depth = np.linalg.norm(pts_list[0], axis=-1)
                    valid_depths = first_depth[first_depth > 0.1]
                    if len(valid_depths) > 0:
                        self.current_metric_scale = 3.0 / np.median(valid_depths)
            else:
                # 1. Get relative scale drift from PanoVGGT
                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                relative_scale = self.current_metric_scale * (0.8 * 1.0 + 0.2 * clipped_scale)
                
                # 2. Gently fuse the Floor Scale if we are confident we are on flat ground
                # This anchors the scene and completely prevents global shrinking/expanding drift
                if floor_scale is not None and floor_conf > 0.4:
                    self.current_metric_scale = 0.9 * relative_scale + 0.1 * floor_scale
                    print(f"  [Metric] DRIFT CORRECTED via Floor | Conf: {floor_conf:.2f}")
                else:
                    self.current_metric_scale = relative_scale
                    print(f"  [Metric] Relied on Pano (No solid floor detected - e.g. Stairs)")
            # =========================================================

            # --- POSE ALIGNMENT ---
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
            
            # --- INTEGRATION ---
            for j in range(self.window_size):
                global_pose = anchor_pose @ canonical_poses[j]
                
                if j >= (0 if self.is_first_window else self.overlap):
                    self.trajectory.append(global_pose[:3, 3])
                
                scaled_pts = pts_list[j] * self.current_metric_scale
                depth_map = np.linalg.norm(scaled_pts, axis=-1)
                
                if j >= (0 if self.is_first_window else self.overlap):
                    self.tsdf.integrate(depth_map, window_frames[j], window_masks[j], global_pose)
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            self.submap_count += 1
            print(f"  [Profile] Cycle Time: {time.time() - t_win_start:.4f} sec")
            
            # Send a fast preview during mapping
            live_pcd = self.tsdf.extract_point_cloud()
            yield None, live_pcd, np.array(self.trajectory), []

        # --- END OF SEQUENCE ---
        print(f"[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        
        # 1. Extract and drastically decimate the mesh
        final_mesh = self.tsdf.extract_mesh(decimation_factor=0.05) 
        
        # 2. THE DENSITY/SIZE FIX: Extract Point Cloud FROM the mesh vertices
        # This gives you an ultra-sharp, tiny point cloud identical to nvblox behavior
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = final_mesh.vertices
        if final_mesh.has_vertex_colors():
            final_pcd.colors = final_mesh.vertex_colors
        
        # 3. CRASH FIX: Return BOTH the mesh and the new sharp point cloud
        yield final_mesh, final_pcd, np.array(self.trajectory), []