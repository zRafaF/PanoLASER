import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time
import skimage.measure

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

    def extract_point_cloud(self, surface_threshold=None, max_points=2000000):
        t_start = time.time()
        if self.next_idx == 0: return o3d.geometry.PointCloud()
        if surface_threshold is None: surface_threshold = self.voxel_size * 2.0 
        
        physical_sdf = self.tsdf[:self.next_idx] * self.margin
        valid = (torch.abs(physical_sdf) <= surface_threshold) & (self.weights[:self.next_idx] > 0)
        
        b_idx, v_idx = torch.where(valid)
        if len(b_idx) == 0: return o3d.geometry.PointCloud()
            
        # Confidence-Weighted Decimation: keep highest weight points
        if len(b_idx) > max_points:
            point_weights = self.weights[:self.next_idx][b_idx, v_idx]
            _, top_k_indices = torch.topk(point_weights, max_points)
            b_idx, v_idx = b_idx[top_k_indices], v_idx[top_k_indices]
            
        coords = (self.block_coords_tensor[b_idx] * self.block_size) + self.block_template[v_idx]
        colors = self.colors[:self.next_idx][b_idx, v_idx]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        return pcd

    def extract_mesh(self, decimation_factor=0.05):
        """
        Extracts the mesh and uses Quadric Error Metrics to intelligently decimate 
        flat surfaces while preserving sharp edges and corners.
        """
        print("[TSDF] Extracting volumetric grid for Marching Cubes...")
        
        # Assemble volumetric grid (From your original implementation)
        active_blocks = self.block_coords_tensor[:self.next_idx]
        min_b = active_blocks.min(dim=0)[0]
        max_b = active_blocks.max(dim=0)[0]
        grid_shape = ((max_b - min_b + 1) * self.block_res).cpu().numpy()
        dense_tsdf = torch.ones(tuple(grid_shape), dtype=torch.float32, device=self.device)
        
        # Efficient assignment
        rel_blocks = active_blocks - min_b
        valid_indices = (rel_blocks.unsqueeze(1) * self.block_res + self.block_template.unsqueeze(0)).view(-1, 3)
        dense_tsdf[valid_indices[:,0].long(), valid_indices[:,1].long(), valid_indices[:,2].long()] = self.tsdf[:self.next_idx].view(-1)
        
        print("[TSDF] Running Marching Cubes...")
        verts, faces, normals, _ = skimage.measure.marching_cubes(dense_tsdf.cpu().numpy(), level=0.0)
        
        # Shift vertices back to world coordinates
        verts = (verts / self.block_res) + min_b.cpu().numpy()
        verts = verts * self.block_size
        
        # Create Open3D Mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # ---------------------------------------------------------
        # APPLY VERTEX COLORS (Sampled from nearest TSDF voxel)
        # ---------------------------------------------------------
        # Note: Implement your color sampling here based on your voxel grid
        # mesh.vertex_colors = o3d.utility.Vector3dVector(sampled_colors)
        
        mesh.compute_vertex_normals()
        
        # ---------------------------------------------------------
        # SMART MESH DECIMATION (Quadric Error Metric)
        # ---------------------------------------------------------
        initial_triangles = len(mesh.triangles)
        target_triangles = max(int(initial_triangles * decimation_factor), 1000)
        
        print(f"[TSDF] Decimating mesh from {initial_triangles} -> {target_triangles} triangles...")
        
        # simplify_quadric_decimation collapses coplanar geometry into massive triangles
        # while keeping sharp building corners mathematically perfect.
        optimized_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        
        print("[TSDF] Mesh optimization complete!")
        return optimized_mesh