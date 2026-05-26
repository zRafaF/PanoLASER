import torch
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .tsdf_volume import FastStaticTSDF  
from .pano_graph import PanoPoseGraph  
from .loop_closure import ImageRetrieval

class StreamingWindowEngineLC:
    def __init__(self, vanilla_engine, window_size=16, overlap=2, device="cuda"):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = device
        
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        
        print("[Engine] Initializing SALAD Loop Closure...")
        self.retriever = ImageRetrieval(input_size=224, device=self.device)
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            
        self.tsdf = FastStaticTSDF(voxel_size=0.02, margin=0.08, max_depth=6.0, device=self.device)
        self.pose_graph = PanoPoseGraph()
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        self.last_integrated_pose = None
        
        self.lc_embeddings = {}
        self.lc_anchor_frames = {}
        self.lc_mid_canonical_poses = {} 
        self.loop_closures = []

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
            
            t_gpu = time.time()
            preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            print(f"  [Profile] VGGT Inference: {time.time() - t_gpu:.4f} sec")
            
            pts_list = preds["points"] 
            poses = preds["poses"]
            
            mid_idx = self.window_size // 2
            mid_frame = window_frames[mid_idx]
            current_emb = self.retriever.get_single_embeding(mid_frame)
            
            self.lc_embeddings[self.submap_count] = current_emb
            self.lc_anchor_frames[self.submap_count] = mid_frame
            
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

            metric_local_poses = []
            for p in poses:
                mp = p.copy()
                mp[:3, 3] *= self.current_metric_scale 
                metric_local_poses.append(mp)
                
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]
            self.lc_mid_canonical_poses[self.submap_count] = canonical_poses[mid_idx]

            if self.is_first_window:
                self.pose_graph.add_prior(self.submap_count, np.eye(4))
            else:
                src_cam_np = np.stack(canonical_poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=1.0)
                
                anchor_guess = np.eye(4)
                anchor_guess[:3, :3] = R_align
                anchor_guess[:3, 3] = t_align
                
                prev_opt_anchor = self.pose_graph.get_optimized_pose(self.submap_count - 1)
                T_rel = np.linalg.inv(prev_opt_anchor) @ anchor_guess
                
                self.pose_graph.add_odometry(self.submap_count - 1, self.submap_count, T_rel, anchor_guess)
                
                best_score = -1.0
                best_match_id = -1
                for old_id, old_emb in self.lc_embeddings.items():
                    if self.submap_count - old_id > 4: 
                        score = torch.nn.functional.cosine_similarity(current_emb, old_emb).item()
                        if score > best_score:
                            best_score = score
                            best_match_id = old_id
                
                if best_score > 0.85: 
                    print(f"  [SLAM] 🟢 LOOP DETECTED: Submap {self.submap_count} -> {best_match_id} (Score: {best_score:.3f})")
                    old_frame = self.lc_anchor_frames[best_match_id]
                    
                    lc_preds = self.engine(np.stack([old_frame, mid_frame]))
                    T_lc_raw = lc_preds["poses"][1] 
                    
                    metric_current_pts = pts_list[mid_idx] * self.current_metric_scale
                    lc_scale = align_cam_pts_irls(
                        torch.from_numpy(lc_preds["points"][1].copy()), 
                        torch.from_numpy(metric_current_pts.copy()), 
                        torch.from_numpy(window_masks[mid_idx].copy())
                    )
                    
                    if 0.1 < lc_scale < 10.0:
                        T_lc_metric = T_lc_raw.copy()
                        T_lc_metric[:3, 3] *= lc_scale
                        
                        # --- [FIX 1]: PROPER UPRIGHTNESS CONSTRAINT ---
                        # Apply a 4x4 matrix sandwich to correctly flip BOTH the 
                        # rotation axes and the translation vector components.
                        if T_lc_metric[1, 1] < 0:
                            flip_4x4 = np.eye(4)
                            flip_4x4[1, 1] = -1
                            flip_4x4[2, 2] = -1
                            T_lc_metric = flip_4x4 @ T_lc_metric @ flip_4x4
                            
                        C_old = self.lc_mid_canonical_poses[best_match_id]
                        C_new = self.lc_mid_canonical_poses[self.submap_count]
                        T_relative_anchors = C_old @ T_lc_metric @ np.linalg.inv(C_new)
                        
                        # --- [FIX 2]: NEUTRALIZE ROTATIONAL TWIST ---
                        # Odometry rotation is highly accurate; VGGT 2-frame rotation is noisy.
                        # Override the LC rotation constraint to perfectly match odometry. 
                        # This forces GTSAM to ONLY use this edge for XYZ translation correction.
                        opt_old = self.pose_graph.get_optimized_pose(best_match_id)
                        opt_new = self.pose_graph.get_optimized_pose(self.submap_count)
                        odom_rel = np.linalg.inv(opt_old) @ opt_new
                        T_relative_anchors[:3, :3] = odom_rel[:3, :3]
                        
                        # --- [FIX 3]: SPATIAL HALLUCINATION GUARD ---
                        odom_dist = np.linalg.norm(opt_new[:3, 3] - opt_old[:3, 3])
                        lc_dist = np.linalg.norm(T_relative_anchors[:3, 3])
                        
                        # Reject the loop closure if VGGT hallucinated a distance > 10 meters off
                        if abs(odom_dist - lc_dist) < 10.0:
                            self.pose_graph.add_loop_closure(best_match_id, self.submap_count, T_relative_anchors)
                            self.loop_closures.append((best_match_id, self.submap_count))
                            print(f"  [SLAM] ✅ LOOP APPLIED (Scale: {lc_scale:.2f}, Drift Corrected: {abs(odom_dist - lc_dist):.2f}m)")
                        else:
                            print(f"  [SLAM] 🟡 LOOP REJECTED (Spatial Hallucination: {lc_dist:.2f}m vs Odom {odom_dist:.2f}m)")
                    else:
                        print(f"  [SLAM] 🟡 LOOP REJECTED (Scale Hallucination: {lc_scale:.2f})")
                
                self.pose_graph.optimize()

            # --- 5. ASYNC TSDF DISPATCH ---
            optimized_anchor = self.pose_graph.get_optimized_pose(self.submap_count)
            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            
            for j in range(self.window_size):
                global_pose = optimized_anchor @ canonical_poses[j]
                
                if global_pose[1, 1] < 0:
                    flip_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    global_pose[:3, :3] = global_pose[:3, :3] @ flip_R

                scaled_pts = pts_list[j] * self.current_metric_scale
                depth_map = np.linalg.norm(scaled_pts, axis=-1)
                
                if j >= (0 if self.is_first_window else self.overlap):
                    batch_depths.append(depth_map)
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(global_pose)
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False

            self.tsdf_future = self.tsdf_executor.submit(
                self._async_tsdf_task, batch_depths, batch_rgbs, batch_masks, batch_poses
            )
            
            self.submap_count += 1
            print(f"  [Profile] Cycle Time: {time.time() - t_win_start:.4f} sec")

        if self.tsdf_future is not None:
            self.tsdf_future.result()

        print(f"[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        
        trajectory = []
        for i in range(self.submap_count):
            pose = self.pose_graph.get_optimized_pose(i)
            trajectory.append(pose[:3, 3]) 
            
        lc_edges = []
        for (from_id, to_id) in self.loop_closures:
            p1 = self.pose_graph.get_optimized_pose(from_id)[:3, 3]
            p2 = self.pose_graph.get_optimized_pose(to_id)[:3, 3]
            lc_edges.append((p1, p2))
            
        return self.tsdf.extract_point_cloud(surface_threshold=0.02), np.array(trajectory), lc_edges