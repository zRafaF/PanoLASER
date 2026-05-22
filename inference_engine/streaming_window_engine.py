import torch
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .tsdf_volume import FastStaticTSDF  
from .pano_graph import PanoPoseGraph  

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=16, overlap=2):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        
        if self.overlap < 2:
            print("WARNING: Kabsch requires overlap >= 2")
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            
        self.tsdf = FastStaticTSDF(voxel_size=0.02, margin=0.08, max_depth=6.0, device=self.device)
        self.pose_graph = PanoPoseGraph()
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        
        self.is_first_window = True
        self.current_metric_scale = 1.0
        self.last_integrated_pose = None
        self.submap_count = 0

    def _is_keyframe(self, current_pose, trans_thresh=0.15, rot_thresh=5.0):
        if self.last_integrated_pose is None:
            return True
        # Both poses are now in strict global metric space (meters)
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
            
            print(f"\n[Streaming Engine] Processing Submap {self.submap_count}...")
            
            # --- 1. GPU INFERENCE ---
            t_gpu_start = time.time()
            preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            print(f"  [Profile] VGGT Submap Inference: {time.time() - t_gpu_start:.4f} sec")
            
            pts_list = preds["points"] 
            poses = preds["poses"]
            
            # --- 2. UPDATE GLOBAL SCALE ---
            if self.is_first_window:
                first_depth = np.linalg.norm(pts_list[0], axis=-1)
                valid_depths = first_depth[first_depth > 0.1]
                if len(valid_depths) > 0:
                    self.current_metric_scale = 3.0 / np.median(valid_depths)
            else:
                # Wait for TSDF thread so variables don't race
                if self.tsdf_future is not None:
                    integ_count, t_time = self.tsdf_future.result()
                    print(f"  [Profile] Background TSDF Thread mapped {integ_count} keyframes.")

                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                # Inertia smoothing
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                self.current_metric_scale *= (0.8 * 1.0 + 0.2 * clipped_scale)

            # --- 3. CONVERT TO CANONICAL LOCAL SPACE ---
            metric_local_poses = []
            for p in poses:
                mp = p.copy()
                mp[:3, 3] *= self.current_metric_scale  # Enforce metric translation
                metric_local_poses.append(mp)
                
            # Shift the entire submap so frame 0 sits at the Origin (I)
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]

            # --- 4. GTSAM GLOBAL OPTIMIZATION ---
            if self.is_first_window:
                anchor_guess = np.eye(4)
                self.pose_graph.add_prior(self.submap_count, anchor_guess)
            else:
                # Align Canonical Local Overlap to Global Previous Overlap
                src_cam_np = np.stack(canonical_poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                
                # Because both spaces are purely metric, scale is explicitly 1.0
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=1.0)
                
                anchor_guess = np.eye(4)
                anchor_guess[:3, :3] = R_align
                anchor_guess[:3, 3] = t_align
                
                prev_opt_anchor = self.pose_graph.get_optimized_pose(self.submap_count - 1)
                
                # Calculate the relative odometry step
                T_rel = np.linalg.inv(prev_opt_anchor) @ anchor_guess
                
                self.pose_graph.add_odometry(
                    from_id=self.submap_count - 1, 
                    to_id=self.submap_count, 
                    relative_mat=T_rel, 
                    initial_estimate_mat=anchor_guess
                )
                
                t_opt = time.time()
                self.pose_graph.optimize()
                print(f"  [Profile] GTSAM Optimization: {time.time() - t_opt:.4f} sec")

            # --- 5. FINALIZE GLOBAL POSES ---
            optimized_anchor = self.pose_graph.get_optimized_pose(self.submap_count)
            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            
            for j in range(self.window_size):
                # Multiply the Canonical Local Pose by the Optimized Global Anchor
                global_pose = optimized_anchor @ canonical_poses[j]
                
                # Uprightness check to prevent the camera from accidentally flipping upside down
                if global_pose[1, 1] < 0:
                    flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    global_pose[:3, :3] = global_pose[:3, :3] @ flip_R

                # Scale the depth map identically to how we scaled the pose
                scaled_pts = pts_list[j] * self.current_metric_scale
                depth_map = np.linalg.norm(scaled_pts, axis=-1)
                
                # Skip duplicate processing of overlap frames
                if j >= (0 if self.is_first_window else self.overlap):
                    batch_depths.append(depth_map)
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(global_pose)
                    
                # Cache the global overlap for the next submap
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False

            # --- 6. ASYNC DISPATCH ---
            self.tsdf_future = self.tsdf_executor.submit(
                self._async_tsdf_task, batch_depths, batch_rgbs, batch_masks, batch_poses
            )
            
            self.submap_count += 1
            print(f"  [Profile] ----- Engine Cycle Time: {time.time() - t_win_start:.4f} sec -----")

        if self.tsdf_future is not None:
            self.tsdf_future.result()

        print(f"[Streaming Engine] Sequence completely mapped in {time.time() - t_seq_start:.4f} sec.")
        return self.tsdf.extract_point_cloud(surface_threshold=0.02)