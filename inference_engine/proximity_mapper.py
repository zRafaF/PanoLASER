import numpy as np
import open3d as o3d
import torch

class ProximityVoxelMap:
    def __init__(self, voxel_size=0.04, max_depth=5.0):
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        
        # Global state buffers
        self.global_pts = np.empty((0, 3), dtype=np.float32)
        self.global_colors = np.empty((0, 3), dtype=np.uint8)
        self.global_dists = np.empty((0,), dtype=np.float32)

    def integrate(self, pts_list, rgbs, masks, poses):
        new_pts = []
        new_colors = []
        new_dists = []

        for i in range(len(poses)):
            local_pts = pts_list[i] 
            rgb = rgbs[i]
            mask = masks[i]
            pose = poses[i]

            if local_pts.ndim == 3:
                local_pts = local_pts.reshape(-1, 3)
                rgb = rgb.reshape(-1, 3)
                mask = mask.reshape(-1)

            # 1. Apply Valid Mask
            valid_mask = mask > 0
            local_pts = local_pts[valid_mask]
            colors = rgb[valid_mask]

            # 2. Calculate true distance from camera center (0,0,0 in local frame)
            # We do this BEFORE the global transform!
            dists = np.linalg.norm(local_pts, axis=-1)

            # 3. Filter by Max Depth Confidence (Ignore noisy distant walls)
            depth_mask = (dists > 0.1) & (dists < self.max_depth)
            local_pts = local_pts[depth_mask]
            colors = colors[depth_mask]
            dists = dists[depth_mask]

            # 4. Transform to Global Space
            R = pose[:3, :3]
            t = pose[:3, 3]
            global_pts = (local_pts @ R.T) + t

            new_pts.append(global_pts)
            new_colors.append(colors)
            new_dists.append(dists)

        if not new_pts:
            return

        # Stack everything together
        all_pts = np.vstack([self.global_pts] + new_pts)
        all_colors = np.vstack([self.global_colors] + new_colors)
        all_dists = np.concatenate([self.global_dists] + new_dists)

        # =======================================================
        # THE PROXIMITY SOLVER ("Closer gets preference")
        # =======================================================
        # 1. Sort all points by distance ASCENDING (closest points first)
        sort_idx = np.argsort(all_dists)
        all_pts = all_pts[sort_idx]
        all_colors = all_colors[sort_idx]
        all_dists = all_dists[sort_idx]

        # 2. Quantize to Voxel Grid (e.g. 5cm blocks)
        voxels = np.round(all_pts / self.voxel_size).astype(np.int32)

        # 3. Keep Unique Voxels
        # Because we sorted by distance, the FIRST occurrence of a voxel coordinate 
        # is guaranteed to be the observation closest to the camera!
        _, unique_idx = np.unique(voxels, axis=0, return_index=True)

        self.global_pts = all_pts[unique_idx]
        self.global_colors = all_colors[unique_idx]
        self.global_dists = all_dists[unique_idx]

    def extract_point_cloud(self):
        pcd = o3d.geometry.PointCloud()
        if len(self.global_pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(self.global_pts)
            pcd.colors = o3d.utility.Vector3dVector(self.global_colors.astype(np.float64) / 255.0)
        return pcd
        
    def extract_poisson_mesh(self, depth=10):
        """Generates a closed, high-quality mesh from the pristine point cloud."""
        pcd = self.extract_point_cloud()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        
        print("[Proximity Map] Generating high-density Poisson Surface Mesh...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        
        # Paint the mesh vertices using our perfect Proximity colors
        mesh_pcd = o3d.geometry.PointCloud()
        mesh_pcd.points = mesh.vertices
        
        # K-Nearest Neighbors to transfer color from point cloud to the new mesh
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        colors = np.asarray(pcd.colors)
        mesh_colors = []
        for v in np.asarray(mesh.vertices):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
            mesh_colors.append(colors[idx[0]])
            
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(mesh_colors))
        return mesh