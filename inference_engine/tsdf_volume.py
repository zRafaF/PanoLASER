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
        # --- GPU Optimization: Track coordinates in a tensor to avoid slow list(keys) conversions ---
        self.block_coords_tensor = torch.zeros((max_blocks, 3), dtype=torch.long, device=self.device)
        self.next_idx = 0
        
        x = torch.arange(self.block_res, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        self.block_template = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).float() * self.voxel_size

    @torch.no_grad()
    def integrate(self, depth_map, rgb_image, mask, pose):
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map.copy()).float().to(self.device)
            mask = torch.from_numpy(mask.copy()).float().to(self.device)
            rgb_image = torch.from_numpy(rgb_image.copy()).float().to(self.device) / 255.0
            pose = torch.from_numpy(pose.copy()).float().to(self.device)
            
        C = pose[:3, 3]
        
        # Surface-Guided Sparse Allocation
        skip = 8
        d_small = depth_map[::skip, ::skip]
        
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
        
        valid = (d_small > 0.1) & (d_small < self.max_depth)
        valid_rays = rays[valid]
        valid_depths = d_small[valid].unsqueeze(-1)
        
        cam_pts = valid_rays * valid_depths
        pose_R = pose[:3, :3]
        world_pts = (cam_pts @ pose_R.T) + C
        
        ray_dirs_world = valid_rays @ pose_R.T
        pts_front = world_pts - ray_dirs_world * self.margin
        pts_back = world_pts + ray_dirs_world * self.margin
        
        all_pts = torch.cat([world_pts, pts_front, pts_back], dim=0)
        
        block_coords = torch.floor(all_pts / self.block_size).long()
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
                # Store natively on GPU
                self.block_coords_tensor[self.next_idx] = valid_blocks[i]
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

    def extract_point_cloud(self, surface_threshold=None, max_points=250000):
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
            
        total_points = len(b_idx)
        if total_points > max_points:
            # --- FIX 1: CONFIDENCE-BASED DECIMATION ---
            # Grab the raw confidence weights of valid surface points
            point_weights = weights[b_idx, v_idx]
            
            # Isolate the indices of the highest-confidence points on the GPU
            _, top_k_indices = torch.topk(point_weights, max_points)
            
            # Filter our coordinates to only keep the top tier data
            b_idx = b_idx[top_k_indices]
            v_idx = v_idx[top_k_indices]
            
        valid_block_coords = self.block_coords_tensor[b_idx]
        valid_local_coords = self.block_template[v_idx]
        
        coords = (valid_block_coords * self.block_size) + valid_local_coords
        colors = self.colors[b_idx, v_idx]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        
        print(f"      [Profile - Extraction] GPU Confidence Extraction: {len(pcd.points)} points in {time.time() - t_start:.4f} sec")
        return pcd

    def extract_mesh(self, surface_threshold=None, poisson_depth=None):
        """Replaces slow CPU Poisson with highly optimized Cython Marching Cubes."""
        t_start = time.time()
        import skimage.measure
        from scipy.spatial import cKDTree
        
        if self.next_idx == 0:
            return o3d.geometry.TriangleMesh()
            
        print("      [Profile - Extraction] Assembling GPU volume for Marching Cubes...")
        # 1. Calculate the active bounding box of the scene
        active_blocks = self.block_coords_tensor[:self.next_idx]
        min_b = active_blocks.min(dim=0)[0]
        max_b = active_blocks.max(dim=0)[0]
        
        grid_dim_blocks = (max_b - min_b) + 1
        grid_shape = (grid_dim_blocks * self.block_res).cpu().numpy()
        
        # 2. Reconstruct dense TSDF locally on the GPU
        dense_tsdf = torch.ones(tuple(grid_shape), dtype=torch.float32, device=self.device)
        
        rel_blocks = active_blocks - min_b
        voxel_coords_global = (rel_blocks.unsqueeze(1) * self.block_res)
        
        x = torch.arange(self.block_res, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        local_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
        
        all_indices = (voxel_coords_global + local_coords.unsqueeze(0)).view(-1, 3)
        
        # Filter out uninitialized empty air blocks to speed up tensor assignment
        B = self.next_idx
        weight_vals = self.weights[:B].view(-1)
        valid_mask = weight_vals > 0
        
        valid_indices = all_indices[valid_mask]
        valid_tsdf = self.tsdf[:B].view(-1)[valid_mask]
        
        # Scatter valid TSDF values into the dense spatial grid
        dense_tsdf[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = valid_tsdf
        
        # Move to CPU for the Cythonized backend
        tsdf_vol = dense_tsdf.cpu().numpy()
        
        print("      [Profile - Extraction] Running Cython Marching Cubes...")
        # Operates directly on the TSDF zero-crossings (No normal estimation required)
        try:
            verts, faces, normals, values = skimage.measure.marching_cubes(tsdf_vol, level=0.0)
        except ValueError:
            return o3d.geometry.TriangleMesh() # Failsafe if volume has no geometry
            
        # Convert local array matrix coordinates back to global 3D world space
        min_b_world = min_b.cpu().numpy() * self.block_size
        verts_world = verts * self.voxel_size + min_b_world
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_world)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        print("      [Profile - Extraction] Colorizing Mesh via Vectorized cKDTree...")
        # Fetch a high-res point cloud to use as a paint palette
        pcd = self.extract_point_cloud(surface_threshold=self.voxel_size * 2, max_points=1500000)
        pcd_verts = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        
        # Vectorized KDTree search assigns colors to 500k vertices in milliseconds
        if len(pcd_verts) > 0:
            tree = cKDTree(pcd_verts)
            _, idx = tree.query(verts_world, k=1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[idx])
        
        print(f"      [Profile - Extraction] Final Mesh extracted in {time.time() - t_start:.4f} sec")
        return mesh