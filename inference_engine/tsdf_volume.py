import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time

class FastStaticTSDF:
    def __init__(self, voxel_size=0.02, margin=0.08, max_depth=6.0, max_blocks=60000, device="cuda"):
        self.device = device
        self.voxel_size = voxel_size
        self.margin = margin
        self.max_depth = max_depth
        
        self.block_res = 16 
        self.block_size = self.block_res * voxel_size
        self.voxels_per_block = self.block_res ** 3
        self.max_blocks = max_blocks
        
        print(f"[TSDF] Allocating Static GPU Pool (~4.5 GB VRAM)...")
        self.tsdf = torch.ones((max_blocks, self.voxels_per_block), dtype=torch.float32, device=self.device)
        self.weights = torch.zeros((max_blocks, self.voxels_per_block), dtype=torch.float32, device=self.device)
        self.colors = torch.zeros((max_blocks, self.voxels_per_block, 3), dtype=torch.float32, device=self.device)
        self.min_dist = torch.full((max_blocks, self.voxels_per_block), float('inf'), dtype=torch.float32, device=self.device)
        
        self.block_hash = {}
        self.next_idx = 0
        
        x = torch.arange(self.block_res, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        self.block_template = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).float() * self.voxel_size

    @torch.no_grad()
    def integrate(self, depth_map, rgb_image, mask, pose):
        t_start = time.time()
        
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map.copy()).float().to(self.device)
            mask = torch.from_numpy(mask.copy()).float().to(self.device)
            rgb_image = torch.from_numpy(rgb_image.copy()).float().to(self.device) / 255.0
            pose = torch.from_numpy(pose.copy()).float().to(self.device)
            
        C = pose[:3, 3]
        
        # --- NEW: SURFACE-GUIDED SPARSE ALLOCATION ---
        # 1. Downsample depth map to speed up raycasting (skip every 8 pixels)
        skip = 8
        d_small = depth_map[::skip, ::skip]
        
        # 2. Generate equirectangular rays matching your panoramic projection
        H, W = d_small.shape
        v_norm = torch.linspace(-1, 1, H, device=self.device)
        u_norm = torch.linspace(-1, 1, W, device=self.device)
        vv, uu = torch.meshgrid(v_norm, u_norm, indexing='ij')
        
        theta = uu * torch.pi
        phi = vv * (torch.pi / 2.0)
        
        X_ray = torch.cos(phi) * torch.sin(theta)
        Y_ray = torch.sin(phi)
        Z_ray = torch.cos(phi) * torch.cos(theta)
        rays = torch.stack([X_ray, Y_ray, Z_ray], dim=-1) 
        
        # 3. Filter to valid points within mapping range
        valid = (d_small > 0.1) & (d_small < self.max_depth)
        valid_rays = rays[valid]
        valid_depths = d_small[valid].unsqueeze(-1)
        
        cam_pts = valid_rays * valid_depths
        
        # 4. Transform to World Space
        pose_R = pose[:3, :3]
        world_pts = (cam_pts @ pose_R.T) + C
        
        # 5. Add margin layers (front/back) to ensure we carve space correctly
        ray_dirs_world = valid_rays @ pose_R.T
        pts_front = world_pts - ray_dirs_world * self.margin
        pts_back = world_pts + ray_dirs_world * self.margin
        
        all_pts = torch.cat([world_pts, pts_front, pts_back], dim=0)
        
        # 6. Hash to unique blocks (No more allocating empty air!)
        block_coords = torch.floor(all_pts / self.block_size).int()
        valid_blocks = torch.unique(block_coords, dim=0)
        
        valid_blocks_cpu = valid_blocks.cpu().numpy()
        
        active_indices = []
        kept_indices = []
        for i, row in enumerate(valid_blocks_cpu):
            k = tuple(row)
            if k not in self.block_hash:
                if self.next_idx >= self.max_blocks:
                    continue
                self.block_hash[k] = self.next_idx
                self.next_idx += 1
            active_indices.append(self.block_hash[k])
            kept_indices.append(i) 
            
        if not active_indices: return
        
        valid_blocks_tensor = valid_blocks[kept_indices]
        B_total = len(active_indices)
        chunk_size = 256
        
        pose_inv = torch.linalg.inv(pose)
        pose_inv_R = pose_inv[:3, :3].T
        pose_inv_t = pose_inv[:3, 3]
        
        depth_tensor = depth_map.unsqueeze(0).unsqueeze(0)
        mask_tensor = mask.unsqueeze(0).unsqueeze(0)
        rgb_tensor = rgb_image.permute(2, 0, 1).unsqueeze(0)

        for i in range(0, B_total, chunk_size):
            chunk_idx = active_indices[i : i + chunk_size]
            chunk_blocks = valid_blocks_tensor[i : i + chunk_size]
            
            B = len(chunk_idx)
            N = self.voxels_per_block
            idx_tensor = torch.tensor(chunk_idx, dtype=torch.long, device=self.device)
            
            old_tsdf = self.tsdf[idx_tensor].view(-1)
            old_w = self.weights[idx_tensor].view(-1)
            old_colors = self.colors[idx_tensor].view(-1, 3)
            old_min_dist = self.min_dist[idx_tensor].view(-1)
            
            offsets = chunk_blocks * self.block_size
            world_coords = (self.block_template.unsqueeze(0) + offsets.unsqueeze(1)).view(B * N, 3)
            
            cam_coords = (world_coords @ pose_inv_R) + pose_inv_t
            X, Y, Z = cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 2]
            r = torch.sqrt(X**2 + Y**2 + Z**2)
            
            phi = torch.asin(torch.clamp(Y / (r + 1e-6), -1.0, 1.0))
            theta = torch.atan2(X, Z)
            
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
            
            sampled_depth = F.grid_sample(depth_tensor, grid, mode='bilinear', align_corners=True).squeeze()
            sampled_mask = F.grid_sample(mask_tensor, grid, mode='nearest', align_corners=True).squeeze()
            sampled_rgb = F.grid_sample(rgb_tensor, grid, mode='bilinear', align_corners=True).squeeze().T
            
            sdf = sampled_depth - r
            valid_mask = (sampled_mask > 0.5) & (sampled_depth > 0.1) & (r < self.max_depth) & (sdf > -self.margin)
            
            tsdf_update = torch.clamp(sdf[valid_mask] / self.margin, -1.0, 1.0)
            
            w_dist = 1.0 / (r[valid_mask] + 0.5) 
            w_carve = torch.where(sdf[valid_mask] > self.margin, 0.1, 1.0).to(self.device)
            new_w = w_dist * w_carve
            
            old_tsdf[valid_mask] = (old_tsdf[valid_mask] * old_w[valid_mask] + tsdf_update * new_w) / (old_w[valid_mask] + new_w)
            old_w[valid_mask] += new_w
            
            closer_mask = r < (old_min_dist - 0.15)
            color_update_mask = valid_mask & closer_mask
            
            if color_update_mask.any():
                old_colors[color_update_mask] = sampled_rgb[color_update_mask]
                old_min_dist[color_update_mask] = r[color_update_mask]
                
            self.tsdf[idx_tensor] = old_tsdf.view(B, N)
            self.weights[idx_tensor] = old_w.view(B, N)
            self.colors[idx_tensor] = old_colors.view(B, N, 3)
            self.min_dist[idx_tensor] = old_min_dist.view(B, N)

    def extract_point_cloud(self, surface_threshold=None):
        t_start = time.time()
        
        if self.next_idx == 0:
            return o3d.geometry.PointCloud()
            
        if surface_threshold is None:
            surface_threshold = self.voxel_size * 2.0 
        
        tsdf_vals = self.tsdf[:self.next_idx]
        weights = self.weights[:self.next_idx]
        
        physical_sdf = tsdf_vals * self.margin
        valid = (torch.abs(physical_sdf) <= surface_threshold) & (weights > 0)
        
        b_idx, v_idx = torch.where(valid)
        if len(b_idx) == 0:
            return o3d.geometry.PointCloud()
            
        keys_list = list(self.block_hash.keys())
        keys_tensor = torch.tensor(keys_list, device=self.device)
        
        valid_block_coords = keys_tensor[b_idx]
        valid_local_coords = self.block_template[v_idx]
        
        coords = (valid_block_coords * self.block_size) + valid_local_coords
        colors = self.colors[b_idx, v_idx]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        
        print(f"      [Profile - Extraction] GPU Vector Extraction: {len(pcd.points)} points in {time.time() - t_start:.4f} sec")
        return pcd

    # --- NEW: MESH EXTRACTION ---
    def extract_mesh(self, surface_threshold=None, poisson_depth=9):
        t_start = time.time()
        pcd = self.extract_point_cloud(surface_threshold)
        
        if len(pcd.points) < 500:
            print("      [Profile - Extraction] Not enough points to compute a valid mesh.")
            return o3d.geometry.TriangleMesh()

        print(f"      [Profile - Extraction] Estimating Normals for {len(pcd.points)} points...")
        # Search radius scaled slightly above voxel size to ensure smooth normal interpolation
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 4.0, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        print("      [Profile - Extraction] Running Poisson Surface Reconstruction...")
        # Depth 9 usually yields sharp room-scale geometry without excessive memory usage
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

        print("      [Profile - Extraction] Cleaning up mesh artifacts...")
        # Poisson algorithms close holes by creating a giant bubble around the scene.
        # We trim away any vertices with low point-cloud density to reveal the true layout.
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Crop strictly to the mapped bounds
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        
        # Color interpolation based on closest points
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_vertices = np.asarray(mesh.vertices)
        pcd_colors = np.asarray(pcd.colors)
        mesh_colors = np.zeros_like(mesh_vertices)
        
        for i in range(len(mesh_vertices)):
            [_, idx, _] = kd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
            mesh_colors[i] = pcd_colors[idx[0]]
            
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

        print(f"      [Profile - Extraction] Mesh created in {time.time() - t_start:.4f} sec")
        return mesh