import torch
import numpy as np
import open3d as o3d
import time
import gc  

from .nvblox_tsdf import NvbloxPanoTSDF
from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .metricfication import estimate_metric_scale_from_floor

class StreamingWindowEngine:
    def __init__(self, vanilla_engine, window_size=16, overlap=2, device="cuda"):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        self.target_camera_height = 1.7 
        
        print("[Engine] Initializing Continuous Nvblox Engine (Fast GPU Fusion)...")
        self.reset()
        
    def reset(self):
        # 1. ONE Continuous Mapper!
        # Because scale is fixed, we can map the whole sequence natively without OOMs
        self.tsdf = NvbloxPanoTSDF(voxel_size_m=0.015, max_depth=5.0, device=self.device)
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        
        gc.collect()
        torch.cuda.empty_cache()

    def process_sequence(self, frames, masks):
        self.reset()
        num_frames = len(frames)
        t_seq_start = time.time()
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            t_win_start = time.time()
            profiler = {}
            
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n==========================================")
            print(f"[Engine] Processing Submap {self.submap_count}...")
            
            # --- PHASE 1: VGGT TRANSFORMER INFERENCE ---
            t0 = time.time()
            with torch.inference_mode():
                preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            profiler["VGGT_Inference"] = time.time() - t0

            pts_list = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["points"]]
            poses = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["poses"]]
            del preds 
            torch.cuda.empty_cache()
            
            # --- PHASE 2: METRICFICATION & ALIGNMENT ---
            t1 = time.time()
            mid_idx = self.window_size // 2
            floor_scale, floor_conf = estimate_metric_scale_from_floor(
                pts_list[mid_idx], target_camera_height=self.target_camera_height
            )
            
            if self.is_first_window:
                if floor_scale is not None:
                    self.current_metric_scale = floor_scale
                    print(f"  > [Metric] Floor Detected | Conf: {floor_conf:.2f} | Scale: {floor_scale:.3f}")
                else:
                    print("  > [Metric] Warning: No floor found. Using median fallback.")
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
            profiler["Scale_&_Pose_Math"] = time.time() - t1
                
            # --- PHASE 3: FAST NATIVE GPU INTEGRATION ---
            t2 = time.time()
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
                
                # Nvblox instantly absorbs overlapping frames! No z-fighting, no manual stitching.
                if j >= (0 if self.is_first_window else self.overlap):
                    self.tsdf.integrate(depth_map, window_frames[j], window_masks[j], tsdf_pose)
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            profiler["Nvblox_Integration"] = time.time() - t2
            
            self.submap_count += 1
            total_time = time.time() - t_win_start
            
            # --- PHASE 4: UI PREVIEW EXTRACTION ---
            t3 = time.time()
            # Extract a sparse preview to stream to VR instantly
            live_pcd = self.tsdf.extract_point_cloud(viz_voxel_scale=4.0)
            if len(live_pcd.points) == 0:
                live_pcd = o3d.geometry.PointCloud()
                live_pcd.points = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
                live_pcd.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
            profiler["UI_Preview_Extraction"] = time.time() - t3
            
            print("  --- Performance Profile ---")
            for k, v in profiler.items():
                print(f"    - {k:<30}: {v:.4f} sec")
            print(f"  >>> Total Cycle Time: {total_time:.4f} sec")
            
            yield None, live_pcd, np.array(self.trajectory), []

        # --- END OF SEQUENCE ---
        print(f"\n[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        print("[Engine] Extracting final global mesh and point cloud from Nvblox...")
        
        # We extract the perfect, undecimated C++ Marching Cubes mesh directly
        final_mesh = self.tsdf.extract_mesh() 
        
        # Get the sharp point cloud from the mesh vertices (naturally small file size)
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = final_mesh.vertices
        if final_mesh.has_vertex_colors():
            final_pcd.colors = final_mesh.vertex_colors
        
        yield final_mesh, final_pcd, np.array(self.trajectory), []