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

    def _build_global_pcd(self, raw_pts, rgb, mask, global_pose, scale=1.0):
        """Filters invalid pixels, applies scale, and transforms into global space."""
        valid = mask.astype(bool)
        # 1. Scale the local geometry points
        pts = raw_pts[valid] * scale
        colors = rgb[valid] / 255.0
        
        # 2. Transform to global space using the strict SE(3) pose
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])
        pts_global = (global_pose @ pts_homo.T).T[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_global)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def process_sequence(self, frames, masks):
        self.reset()
        
        # 1. Generate all local PCDs and initial poses
        pcds = []
        poses = []
        for i in range(len(frames)):
            preds = self.engine([frames[i]]) # Single frame inference
            pts = preds["points"][0]
            pose = preds["poses"][0]
            
            pcd = self._build_global_pcd(pts, frames[i], masks[i], pose, scale=1.0)
            pcds.append(pcd)
            poses.append(pose)
            
        # 2. Build Pose Graph for Global Optimization
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.eye(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        
        for i in range(1, len(pcds)):
            # Use Point-to-Plane ICP for fine-grained locking
            icp = o3d.pipelines.registration.registration_icp(
                pcds[i], pcds[i-1], 2.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            odometry = odometry @ icp.transformation
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            
            # Add edge
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i-1, i, icp.transformation, uncertain=False))
            
        # 3. Optimize Global Consistency
        option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=2.0)
        o3d.pipelines.registration.global_optimization(
            pose_graph, 
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), 
            option
        )
        
        # 4. Integrate
        for i in range(len(pcds)):
            pcds[i].transform(pose_graph.nodes[i].pose)
            self.global_pcd += pcds[i]
            
        return self.global_pcd.voxel_down_sample(voxel_size=0.05)