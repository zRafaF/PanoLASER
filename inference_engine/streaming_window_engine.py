import torch
import numpy as np
import open3d as o3d
import time

from .inference_utils import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch
from .tsdf_volume import SphericalTSDFVolume  # Import our new GPU Volume

class PanoStreamingEngine:
    def __init__(self, vanilla_engine, window_size=3, overlap=2):
        self.engine = vanilla_engine
        self.window_size = window_size
        self.overlap = overlap
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.overlap < 2:
            print("WARNING: Kabsch requires an overlap of >= 2 to guarantee a stable 3D rotation.")
        self.reset()
        
    def reset(self):
        """Clears the temporal memory and initializes a new TSDF Volume."""
        # [FIX 1]: Expand the bounds massively to prevent clipping. 
        # Now a 20m x 8m x 20m tracking volume.
        vol_bounds = [[-10.0, 10.0], [-4.0, 4.0], [-10.0, 10.0]]
        self.tsdf = SphericalTSDFVolume(vol_bounds=vol_bounds, voxel_size=0.02, margin=0.08, device=self.device)
        
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
        # In the camera pose matrix, this is stored in the 2nd column (index 1).
        up_vector_world = candidate_pose[:3, 1] 
        
        # If the Y-component is negative, the camera thinks 'up' is 'down'
        if up_vector_world[1] < 0:
            print("  -> [Geometry Warning] Detected upside-down frame, applying 180° flip.")
            flip_R = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            candidate_pose[:3, :3] = candidate_pose[:3, :3] @ flip_R
            
        return candidate_pose

    def process_sequence(self, frames, masks):
        """Processes a sequence using Exact Odometry Chaining and TSDF Integration."""
        self.reset()
        num_frames = len(frames)
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]
            
            print(f"\n[Streaming Engine] Processing Window {i}...")
            
            t_gpu_start = time.time()
            preds = self.engine(window_frames)
            print(f"  [Profile] GPU Neural Network Inference: {time.time() - t_gpu_start:.4f} sec")
            
            pts_list = preds["points"] 
            poses = preds["poses"]
            
            if self.is_first_window:
                self.prev_overlap_global_poses = []
                
                # [FIX 2]: Calculate the inverse of the first pose to act as our leveling anchor
                self.world_anchor = np.linalg.inv(poses[0])
                
                for j in range(self.window_size):
                    # Map the raw network pose to our perfectly level origin
                    aligned_pose = self.world_anchor @ poses[j]
                    
                    depth_map = np.linalg.norm(pts_list[j], axis=-1)
                    
                    self.tsdf.integrate(
                        depth_map=depth_map, 
                        rgb_image=window_frames[j], 
                        mask=window_masks[j], 
                        pose=aligned_pose  # Use the perfectly leveled pose!
                    )
                    # Save the aligned poses so subsequent windows stitch to the leveled map
                    self.prev_overlap_global_poses.append(aligned_pose)
                
                self.prev_overlap_raw_pts = pts_list[-self.overlap:]
                self.prev_overlap_global_poses = self.prev_overlap_global_poses[-self.overlap:]
                self.is_first_window = False
                
            else:
                t_align_start = time.time()
                # 1. Calculate Scale Drift
                scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                
                # 2. Rigid Odometry Stitching (Kabsch Alignment)
                # Map the current local overlap to the previous global overlap
                src_cam_np = np.stack(poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=scale_diff)
                print(f"  [Profile] Sim3 Kabsch Alignment took {time.time() - t_align_start:.4f} sec")
                
                new_global_poses = []
                for j in range(self.window_size):
                    # Transform the local network pose into the unified global map space
                    global_pose = self._apply_sim3_to_pose(poses[j], R_align, t_align, scale_diff)
                    new_global_poses.append(global_pose)
                    
                    # Only map the NEW frames (ignore the overlap ones we already mapped)
                    if j >= self.overlap:
                        depth_map = np.linalg.norm(pts_list[j], axis=-1)
                        
                        self.tsdf.integrate(
                            depth_map=depth_map, 
                            rgb_image=window_frames[j], 
                            mask=window_masks[j], 
                            pose=global_pose,  # Use the ALIGNED pose!
                            scale=scale_diff
                        )
                
                # Save the end of this newly aligned window to be the target for the next one
                self.prev_overlap_raw_pts = pts_list[-self.overlap:]
                self.prev_overlap_global_poses = new_global_poses[-self.overlap:]

        print("[Streaming Engine] Sequence processing complete.")
        
        # 3. RAZOR SHARP EXTRACTION: Only extract voxels within 2cm of the exact surface crossing
        final_pcd = self.tsdf.extract_point_cloud(surface_threshold=0.02)
        
        return final_pcd