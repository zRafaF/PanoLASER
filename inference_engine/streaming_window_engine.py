import torch
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor

from .nvblox_tsdf import NvbloxPanoTSDF
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
        
        # User Mutable Tuning Parameters
        self.max_depth = 4.5
        self.voxel_size = 0.02
        self.min_translation_m = 0.10
        
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        self.tsdf = None
        
        print("[Engine] Initializing Dynamic C++ Nvblox Keyframe Engine...")
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            self.tsdf_future = None

        self.last_mesh = None
        self.last_pcd = None
            
        self.tsdf = NvbloxPanoTSDF(voxel_size_m=self.voxel_size, max_depth=self.max_depth, crop_margin=24, device=self.device)
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.full_poses = []
        self.processed_indices = []

        self.kf_rgbs = []
        self.kf_depths = []
        self.kf_masks = []
        self.kf_poses = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        
        self.last_integrated_position = None

    def _async_tsdf_task(self, depth_maps, rgb_frames, masks, poses):
        for j in range(len(poses)):
            self.tsdf.integrate(depth_maps[j], rgb_frames[j], masks[j], poses[j])
        torch.cuda.synchronize()

    @torch.no_grad()
    def _apply_pytorch_colors(self, vertices_np, normals_np, batch_rgbs, batch_depths, batch_masks, batch_poses):
        """
        Projects mesh vertices back into panoramic keyframes.
        Features: Distance Attenuation, Angle Alignment, Z-Buffer Occlusion, and Deadzone Masking.
        """
        num_pts = vertices_np.shape[0]
        if num_pts == 0 or len(batch_rgbs) == 0:
            return np.zeros((num_pts, 3))

        vertices = torch.from_numpy(vertices_np).float().to(self.device)
        normals = torch.from_numpy(normals_np).float().to(self.device)

        best_scores = torch.full((num_pts,), -1.0, device=self.device)
        final_colors = torch.zeros((num_pts, 3), device=self.device)

        for rgb_np, depth_np, mask_np, pose_np in zip(batch_rgbs, batch_depths, batch_masks, batch_poses):
            # 1. Prepare Tensors (1, Channels, H, W)
            img_t = torch.from_numpy(rgb_np).float().to(self.device) / 255.0
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)
            
            depth_t = torch.from_numpy(depth_np).float().to(self.device).unsqueeze(0).unsqueeze(0)
            mask_t = torch.from_numpy(mask_np).float().to(self.device).unsqueeze(0).unsqueeze(0)

            # 2. Camera pose math
            pose_t = torch.from_numpy(pose_np).float().to(self.device)
            R_w_c = pose_t[:3, :3]
            t_w_c = pose_t[:3, 3]

            V_c = (vertices - t_w_c) @ R_w_c 
            X, Y, Z = V_c[:, 0], V_c[:, 1], V_c[:, 2]
            dist = torch.sqrt(X**2 + Y**2 + Z**2)

            # 3. Convert to Equirectangular UVs [-1, 1]
            theta = torch.atan2(X, Z)
            phi = torch.asin(torch.clamp(Y / (dist + 1e-6), -1.0, 1.0))
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0) 

            # 4. Simultaneous Grid Sampling
            sampled_colors = torch.nn.functional.grid_sample(img_t, grid, mode='bilinear', align_corners=True).squeeze().T
            sampled_depths = torch.nn.functional.grid_sample(depth_t, grid, mode='nearest', align_corners=True).squeeze()
            sampled_masks = torch.nn.functional.grid_sample(mask_t, grid, mode='nearest', align_corners=True).squeeze()

            # --- THE FIXES ---
            
            # A. Mask Check: Did we hit the black pole/sky?
            valid_mask_hit = sampled_masks > 0.5
            
            # B. Z-Buffer Occlusion: Is the vertex hiding behind a surface?
            # We add a 15cm tolerance (0.15) to account for mesh smoothing/quantization
            depth_tolerance = 0.15 
            is_visible = dist <= (sampled_depths + depth_tolerance)

            # C. Score Calculation
            view_dirs_w = (t_w_c - vertices) / (dist.unsqueeze(1) + 1e-6)
            dot_prod = (view_dirs_w * normals).sum(dim=1)
            
            # Must be front-facing AND unoccluded AND inside the valid mask
            valid_condition = (dist > 0.1) & valid_mask_hit & is_visible & (dot_prod > 0)

            score = torch.where(
                valid_condition,
                dot_prod / (dist**2 + 1e-6),
                torch.tensor(-1.0, device=self.device)
            )

            # Winner takes all
            update_mask = score > best_scores
            best_scores[update_mask] = score[update_mask]
            final_colors[update_mask] = sampled_colors[update_mask]

        return final_colors.cpu().numpy()

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
            
            t0 = time.time()
            if self.tsdf_future is not None:
                self.tsdf_future.result()
                self.tsdf_future = None
            profiler["TSDF_Sync"] = time.time() - t0
            
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
                    current_pos = global_pose[:3, 3]
                    self.trajectory.append(current_pos)
                    
                    self.processed_indices.append(i + j)
                    self.full_poses.append(global_pose)
                    
                    should_integrate = False
                    if self.last_integrated_position is None:
                        should_integrate = True
                    else:
                        dist = np.linalg.norm(current_pos - self.last_integrated_position)
                        if dist >= self.min_translation_m:
                            should_integrate = True
                            
                    if should_integrate:
                        self.last_integrated_position = current_pos.copy()
                        
                        tsdf_pose = global_pose.copy()
                        if tsdf_pose[1, 1] < 0:
                            flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                            tsdf_pose[:3, :3] = tsdf_pose[:3, :3] @ flip_R

                        scaled_pts = pts_list[j] * self.current_metric_scale
                        depth_map = np.linalg.norm(scaled_pts, axis=-1)
                        
                        # Local batches for C++ Nvblox
                        batch_depths.append(depth_map)
                        batch_rgbs.append(window_frames[j])
                        batch_masks.append(window_masks[j])
                        batch_poses.append(tsdf_pose) 
                        
                        # Global tracking for PyTorch Shader
                        self.kf_depths.append(depth_map)
                        self.kf_rgbs.append(window_frames[j])
                        self.kf_masks.append(window_masks[j])
                        self.kf_poses.append(tsdf_pose) 
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            profiler["Scale_&_Pose_Math"] = time.time() - t2

            t3 = time.time()
            if len(batch_poses) > 0:
                print(f"  > [TSDF] Sending {len(batch_poses)} Keyframes to C++ Background Mapper...")
                self.tsdf_future = self.tsdf_executor.submit(
                    self._async_tsdf_task, batch_depths, batch_rgbs, batch_masks, batch_poses
                )
            profiler["Nvblox_Enqueue"] = time.time() - t3
            
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
            
            if self.submap_count % 3 == 0:
                self.last_mesh = self.tsdf.extract_mesh()
                
                # Apply PyTorch Coloring Shader
                if self.last_mesh is not None and len(self.last_mesh.vertices) > 0:
                    t_color_start = time.time()
                    
                    self.last_mesh.compute_vertex_normals()
                    
                    colored_vertices = self._apply_pytorch_colors(
                        np.asarray(self.last_mesh.vertices),
                        np.asarray(self.last_mesh.vertex_normals),
                        self.kf_rgbs,
                        self.kf_depths,
                        self.kf_masks,
                        self.kf_poses
                    )
                    
                    self.last_mesh.vertex_colors = o3d.utility.Vector3dVector(colored_vertices)
                    print(f"  > [Coloring] PyTorch Raycasting applied to {len(self.last_mesh.vertices)} vertices in {time.time() - t_color_start:.4f} sec")

                self.last_pcd = o3d.geometry.PointCloud()
                self.last_pcd.points = self.last_mesh.vertices
                if self.last_mesh.has_vertex_colors():
                    self.last_pcd.colors = self.last_mesh.vertex_colors

            yield_mesh = self.last_mesh
            yield_pcd = self.last_pcd

            if yield_mesh is None or len(yield_mesh.vertices) == 0:
                yield_mesh = o3d.geometry.TriangleMesh()
                yield_pcd = o3d.geometry.PointCloud()
                
                if len(self.trajectory) > 0:
                    yield_pcd.points = o3d.utility.Vector3dVector(np.array(self.trajectory))
                    yield_pcd.paint_uniform_color([1.0, 0.0, 0.0]) 
                else:
                    yield_pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
                
                yield_mesh.vertices = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0]]))
                yield_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

            yield yield_mesh, yield_pcd, np.array(self.trajectory), []

        if self.tsdf_future is not None:
            self.tsdf_future.result()

        print("[Engine] Final Sequence: Extracting Unified Global Mesh...")
        final_mesh = self.tsdf.extract_mesh()
        
        # Apply Final Global Coloring (FIXED)
        if final_mesh is not None and len(final_mesh.vertices) > 0:
            final_mesh.compute_vertex_normals()
            final_colors = self._apply_pytorch_colors(
                np.asarray(final_mesh.vertices),
                np.asarray(final_mesh.vertex_normals),
                self.kf_rgbs,
                self.kf_depths,
                self.kf_masks,
                self.kf_poses
            )
            final_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)
        
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = final_mesh.vertices
        if final_mesh.has_vertex_colors():
            final_pcd.colors = final_mesh.vertex_colors

        print(f"\n[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        yield final_mesh, final_pcd, np.array(self.trajectory), []