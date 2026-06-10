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
        
        self.max_depth = 4.5
        self.voxel_size = 0.02
        self.min_translation_m = 0.10
        
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        self.tsdf = None
        
        print("[Engine] Initializing PyTorch Shaded TSDF Engine...")
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
        Batched and Chunked implementation. 
        Projects vertices into all keyframes simultaneously to avoid Python loop overhead.
        """
        num_pts = vertices_np.shape[0]
        if num_pts == 0 or len(batch_rgbs) == 0:
            return np.zeros((num_pts, 3))

        B = len(batch_rgbs)
        device = self.device

        # Pre-load entire batch to GPU 
        imgs_t = torch.from_numpy(np.stack(batch_rgbs)).float().to(device) / 255.0
        imgs_t = imgs_t.permute(0, 3, 1, 2) # (B, 3, H, W)
        
        depths_t = torch.from_numpy(np.stack(batch_depths)).float().to(device).unsqueeze(1) # (B, 1, H, W)
        masks_t = torch.from_numpy(np.stack(batch_masks)).float().to(device).unsqueeze(1) # (B, 1, H, W)
        
        poses_t = torch.from_numpy(np.stack(batch_poses)).float().to(device)
        R_w_c = poses_t[:, :3, :3] # (B, 3, 3)
        t_w_c = poses_t[:, :3, 3]  # (B, 3)

        vertices = torch.from_numpy(vertices_np).float().to(device)
        normals = torch.from_numpy(normals_np).float().to(device)

        final_colors = torch.zeros((num_pts, 3), device=device)
        
        # Process in chunks to prevent VRAM explosion when batching B cameras
        CHUNK_SIZE = 500_000 
        
        for start_idx in range(0, num_pts, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, num_pts)
            v_chunk = vertices[start_idx:end_idx] # (C, 3)
            n_chunk = normals[start_idx:end_idx]  # (C, 3)
            C_size = v_chunk.shape[0]

            # Vectorized projection: (P - T) @ R
            diff = v_chunk.unsqueeze(0) - t_w_c.unsqueeze(1) # (B, C, 3)
            V_c = torch.bmm(diff, R_w_c) # (B, C, 3)
            
            X, Y, Z = V_c[..., 0], V_c[..., 1], V_c[..., 2]
            dist = torch.sqrt(X**2 + Y**2 + Z**2) # (B, C)

            theta = torch.atan2(X, Z)
            phi = torch.asin(torch.clamp(Y / (dist + 1e-6), -1.0, 1.0))
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2) # (B, C, 1, 2)

            sampled_colors = torch.nn.functional.grid_sample(imgs_t, grid, mode='bilinear', align_corners=False).squeeze(3) # (B, 3, C)
            sampled_depths = torch.nn.functional.grid_sample(depths_t, grid, mode='nearest', align_corners=False).squeeze(3).squeeze(1) # (B, C)
            sampled_masks = torch.nn.functional.grid_sample(masks_t, grid, mode='nearest', align_corners=False).squeeze(3).squeeze(1) # (B, C)

            valid_mask_hit = sampled_masks > 0.5
            is_not_black = sampled_colors.sum(dim=1) > 0.15 # (B, C)
            is_visible = dist <= (sampled_depths + 0.15)
            
            view_dirs_w = -diff / (dist.unsqueeze(-1) + 1e-6) # (B, C, 3)
            dot_prod = (view_dirs_w * n_chunk.unsqueeze(0)).sum(dim=-1) # (B, C)
            
            lens_confidence = torch.cos(phi) # (B, C)
            
            valid_condition = (dist > 0.1) & valid_mask_hit & is_visible & (dot_prod > 0) & is_not_black

            score = torch.where(
                valid_condition,
                (dot_prod * lens_confidence) / (dist**2 + 1e-6),
                torch.tensor(-1.0, device=device)
            ) # (B, C)

            # Pick the best camera for each point
            max_scores, best_cam_idx = torch.max(score, dim=0) # (C,), (C,)
            
            update_mask = max_scores > -1.0
            if update_mask.any():
                valid_cams = best_cam_idx[update_mask]
                valid_pts = torch.arange(C_size, device=device)[update_mask]
                
                best_colors = sampled_colors[valid_cams, :, valid_pts] # (N_valid, 3)
                final_colors[start_idx:end_idx][update_mask] = best_colors

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
                        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        batch_depths.append(depth_map)
                        batch_rgbs.append(window_frames[j])
                        batch_masks.append(window_masks[j])
                        batch_poses.append(tsdf_pose) 
                        
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
            
            pt_alloc, pt_res, sys_used, sys_total = self.vram_tracker.stop()
            total_time = time.time() - t_win_start
            
            print("  --- Performance Profile ---")
            for k, v in sorted(profiler.items()):
                print(f"    - {k:<25}: {v:.4f} sec")
            print(f"  >>> Total Cycle Time: {total_time:.4f} sec")
            print("  --- GPU Memory (VRAM) ---")
            print(f"    - PyTorch Peak Alloc : {pt_alloc:.2f} GB")
            print(f"    - PyTorch Peak Rsvd  : {pt_res:.2f} GB")
            print(f"    - True System VRAM   : {sys_used:.2f} GB / {sys_total:.2f} GB")

            self.submap_count += 1
            
            if self.submap_count % 3 == 0:
                # Lock background thread before requesting geometry
                if self.tsdf_future is not None:
                    self.tsdf_future.result()
                    self.tsdf_future = None
                    
                self.last_mesh = self.tsdf.extract_mesh()
                
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
                if self.last_mesh is not None and len(self.last_mesh.vertices) > 0:
                    # Filter out unpainted vertices so they don't look like floating dust
                    valid_color_mask = colored_vertices.sum(axis=1) > 0.0
                    filtered_points = np.asarray(self.last_mesh.vertices)[valid_color_mask]
                    filtered_colors = colored_vertices[valid_color_mask]
                    
                    self.last_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                    self.last_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

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
        if final_mesh is not None and len(final_mesh.vertices) > 0:
            valid_color_mask = final_colors.sum(axis=1) > 0.0
            filtered_points = np.asarray(final_mesh.vertices)[valid_color_mask]
            filtered_colors = final_colors[valid_color_mask]
            
            final_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            final_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        print(f"\n[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        yield final_mesh, final_pcd, np.array(self.trajectory), []