import torch
import numpy as np
import open3d as o3d
import time
import gc  
from concurrent.futures import ThreadPoolExecutor

from .tsdf_volume import FastStaticTSDF
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
        
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        self.tsdf = None
        
        print("[Engine] Initializing PyTorch Engine with Streaming Optimization & Profiling...")
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            del self.tsdf_future
            self.tsdf_future = None
            
        self.global_pcd = o3d.geometry.PointCloud()
        self.global_mesh = o3d.geometry.TriangleMesh()
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        
        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0
        
        if self.tsdf is not None:
            del self.tsdf
            self.tsdf = None
            
        gc.collect()
        torch.cuda.empty_cache()

    def _async_tsdf_task(self, tsdf_instance, depth_maps, rgb_frames, masks, poses):
        """Runs the TSDF integration and local optimization in the background thread."""
        for j in range(len(poses)):
            tsdf_instance.integrate(depth_maps[j], rgb_frames[j], masks[j], poses[j])
        torch.cuda.synchronize()
        
        # 1. Extract raw local structures
        local_pcd = tsdf_instance.extract_point_cloud()
        local_mesh = tsdf_instance.extract_mesh() # Now you don't decimate heavily here!
        
        # 2. LOCAL POINT CLOUD OPTIMIZATION (Statistical Outlier Removal)
        # Removes floating noise and grey artifacts before they reach the global map
        if len(local_pcd.points) > 50:
            local_pcd, ind = local_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return local_mesh, local_pcd

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
            
            # --- PHASE 1: RECLAIM & GLOBAL OPTIMIZATION ---
            t0 = time.time()
            if self.tsdf_future is not None:
                local_mesh, local_pcd = self.tsdf_future.result()
                profiler["TSDF_Wait"] = time.time() - t0
                
                t1 = time.time()
                # ICP Alignment
                local_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                if len(self.global_pcd.points) > 0:
                    self.global_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    icp_result = o3d.pipelines.registration.registration_icp(
                        local_pcd, self.global_pcd, 
                        max_correspondence_distance=0.15,
                        init=np.eye(4),
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    )
                    local_pcd.transform(icp_result.transformation)
                    local_mesh.transform(icp_result.transformation)
                profiler["CPU_ICP_Align"] = time.time() - t1
                
                t2 = time.time()
                # Append chunks
                self.global_pcd += local_pcd
                self.global_mesh += local_mesh
                
                # GLOBAL STREAMING OPTIMIZATION
                # 1. Point Cloud Zipper
                self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.01)
                
                # 2. Mesh Vertex Clustering (Extremely fast, preserves small objects, sews chunks together)
                self.global_mesh = self.global_mesh.simplify_vertex_clustering(
                    voxel_size=0.01, 
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
                profiler["Global_Mesh_PCD_Optimization"] = time.time() - t2
                
                # Cleanup
                t3 = time.time()
                del self.tsdf_future, self.tsdf
                self.tsdf_future = None
                self.tsdf = None
                gc.collect()
                torch.cuda.empty_cache()
                self.tsdf_executor.submit(lambda: None).result()
                profiler["Garbage_Collection"] = time.time() - t3
            
            # --- PHASE 2: VGGT TRANSFORMER INFERENCE ---
            t4 = time.time()
            with torch.inference_mode():
                preds = self.engine(window_frames)
            torch.cuda.synchronize()  
            profiler["VGGT_Inference"] = time.time() - t4

            t5 = time.time()
            pts_list = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["points"]]
            poses = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in preds["poses"]]
            del preds 
            torch.cuda.empty_cache()
            
            # --- PHASE 3: METRICFICATION & ALIGNMENT ---
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
                batch_depths.append(np.linalg.norm(scaled_pts, axis=-1))
                batch_rgbs.append(window_frames[j])
                batch_masks.append(window_masks[j])
                batch_poses.append(tsdf_pose) 
                    
                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            profiler["Scale_&_Pose_Math"] = time.time() - t5

            # --- PHASE 4: ASYNC INTEGRATION ---
            t6 = time.time()
            self.tsdf = FastStaticTSDF(voxel_size=0.01, max_depth=5.0, device=self.device)
            self.tsdf_future = self.tsdf_executor.submit(
                self._async_tsdf_task, self.tsdf, batch_depths, batch_rgbs, batch_masks, batch_poses
            )
            profiler["TSDF_Spawn"] = time.time() - t6
            
            self.submap_count += 1
            total_time = time.time() - t_win_start
            
            # --- PRINT PROFILING REPORT ---
            print("  --- Performance Profile ---")
            for k, v in profiler.items():
                print(f"    - {k:<30}: {v:.4f} sec")
            print(f"  >>> Total Cycle Time: {total_time:.4f} sec")
            
            # YIELD LIVE STREAM (Safeguarded for first frame)
            safe_pcd = self.global_pcd
            safe_mesh = self.global_mesh
            if len(safe_pcd.points) == 0:
                safe_pcd = o3d.geometry.PointCloud()
                safe_pcd.points = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
                safe_mesh = o3d.geometry.TriangleMesh()
                
            yield safe_mesh, safe_pcd, np.array(self.trajectory), []

        # --- END OF SEQUENCE ---
        if self.tsdf_future is not None:
            local_mesh, local_pcd = self.tsdf_future.result()
            self.global_pcd += local_pcd
            self.global_mesh += local_mesh
            
            self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.01)
            self.global_mesh = self.global_mesh.simplify_vertex_clustering(
                voxel_size=0.01, contraction=o3d.geometry.SimplificationContraction.Average
            )
            
            del self.tsdf_future, self.tsdf
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\\n[Engine] Sequence mapped in {time.time() - t_seq_start:.4f} sec.")
        yield self.global_mesh, self.global_pcd, np.array(self.trajectory), []