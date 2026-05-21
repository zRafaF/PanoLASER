import torch
import numpy as np
import open3d as o3d

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=3, overlap=2):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        
        if self.overlap < 2:
            print("WARNING: Kabsch requires an overlap of >= 2 to guarantee a stable 3D rotation.")
        self.reset()
        
    def reset(self):
        """Clears the temporal memory for a new video sequence."""
        self.global_pcd = o3d.geometry.PointCloud()
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.prev_overlap_masks = []
        self.is_first_window = True

    def _apply_sim3_to_pose(self, local_pose, R_align, t_align, scale):
        """
        Applies Sim(3) with an Uprightness Constraint to prevent floor/ceiling flipping.
        """
        # 1. Scale ONLY the translation component
        L_scaled = local_pose.copy()
        L_scaled[:3, 3] *= scale

        # 2. Construct Rigid Alignment
        T_align = np.eye(4)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = t_align

        # 3. Calculate candidate pose
        candidate_pose = T_align @ L_scaled
        
        # 4. UP-VECTOR CONSISTENCY CHECK
        # The 'Up' vector in world space is (0, 1, 0). 
        # In the camera pose matrix, this is stored in the 2nd column (index 1) of the rotation part.
        up_vector_world = candidate_pose[:3, 1] 
        
        # If the Y-component is negative, the camera thinks 'up' is 'down'
        if up_vector_world[1] < 0:
            print("  -> [Geometry Warning] Detected upside-down frame, applying 180° flip.")
            # Apply a 180-degree rotation around the X-axis to flip the frame
            flip_R = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            candidate_pose[:3, :3] = candidate_pose[:3, :3] @ flip_R
            
        return candidate_pose

    def _build_global_pcd(self, raw_pts, rgb, mask, global_pose, scale=1.0, max_depth=10.0):
        """Filters invalid pixels, applies scale, cleans outliers, and transforms to global space."""
        # 1. Depth Confidence Thresholding (Heuristic)
        # Calculate distances from the origin (camera center)
        distances = np.linalg.norm(raw_pts, axis=1)
        valid = mask.astype(bool) & (distances < max_depth)
        
        pts = raw_pts[valid] * scale
        colors = rgb[valid] / 255.0
        
        # 2. Transform to global space
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])
        pts_global = (global_pose @ pts_homo.T).T[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_global)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 3. Statistical Outlier Removal
        # nb_neighbors: how many neighbors to analyze
        # std_ratio: lower means more aggressive filtering
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd

    def process_sequence(self, frames, masks):
        """Processes a sequence using Exact Odometry Chaining & IRLS Scale Alignment."""
        self.reset()
        num_frames = len(frames)
        
        # We loop until we cannot form a full window anymore
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n[Streaming Engine] Processing Window {i}...")
            preds = self.engine(window_frames)
            pts_list = preds["points"]
            poses = preds["poses"]
            
            if self.is_first_window:
                for j in range(self.window_size):
                    pcd = self._build_global_pcd(pts_list[j], window_frames[j], window_masks[j], poses[j])
                    self.global_pcd += pcd
                
                self.global_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
                
                self.prev_overlap_raw_pts = pts_list[-self.overlap:]
                self.prev_overlap_global_poses = list(poses[-self.overlap:])
                self.is_first_window = False
                
            else:
                # 1. IRLS Scale Drift Correction
                # Use .copy() to ensure the array is writable for PyTorch
                scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                
                # 2. Point-to-Plane ICP
                # j starts from overlap, not 0, to avoid re-processing the overlap frame
                for j in range(self.overlap, self.window_size):
                    pcd = self._build_global_pcd(pts_list[j], window_frames[j], window_masks[j], poses[j], scale=scale_diff)
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
                    
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd, self.global_pcd, 2.0, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    )
                    pcd.transform(reg.transformation)
                    self.global_pcd += pcd
                    self.global_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
                
                self.prev_overlap_raw_pts = pts_list[-self.overlap:]

        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=0.02)
        print("[Streaming Engine] Sequence processing complete.")
        return self.global_pcd