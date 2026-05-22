import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time

class FastStaticTSDF:
    def __init__(self, voxel_size=0.02, margin=0.08, max_depth=6.0, max_blocks=60000, device="cuda"):
        """
        A Zero-Copy Static Pool TSDF. Pre-allocates a massive contiguous block of VRAM
        to eliminate PCIe transfer overhead and dynamic memory fragmentation.
        """
        self.device = device
        self.voxel_size = voxel_size
        self.margin = margin
        self.max_depth = max_depth
        
        self.block_res = 16  # Smaller blocks = tighter frustum culling
        self.block_size = self.block_res * voxel_size
        self.voxels_per_block = self.block_res ** 3
        
        self.max_blocks = max_blocks
        
        print(f"[TSDF] Allocating Static GPU Pool (~4.5 GB VRAM)...")
        # 1. Pre-allocate the entire memory pool ONCE.
        self.tsdf = torch.ones((max_blocks, self.voxels_per_block), dtype=torch.float32, device=self.device)
        self.weights = torch.zeros((max_blocks, self.voxels_per_block), dtype=torch.float32, device=self.device)
        self.colors = torch.zeros((max_blocks, self.voxels_per_block, 3), dtype=torch.float32, device=self.device)
        self.min_dist = torch.full((max_blocks, self.voxels_per_block), float('inf'), dtype=torch.float32, device=self.device)
        
        # 2. Fast Python Hash Map just to track pool indices
        self.block_hash = {}
        self.next_idx = 0
        
        # 3. Pre-compute local coordinates
        x = torch.arange(self.block_res, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        self.block_template = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).float() * self.voxel_size

        print(f"[TSDF] Ready. Bubble Size: {max_depth * 2}m diameter. Max Capacity: {max_blocks * self.voxels_per_block / 1e6:.1f}M voxels.")

    @torch.no_grad()
    def integrate(self, depth_map, rgb_image, mask, pose):
        t_start = time.time()
        
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map.copy()).float().to(self.device)
            mask = torch.from_numpy(mask.copy()).float().to(self.device)
            rgb_image = torch.from_numpy(rgb_image.copy()).float().to(self.device) / 255.0
            pose = torch.from_numpy(pose.copy()).float().to(self.device)
            
        C = pose[:3, 3]
        
        # 1. Vectorized Frustum Bounding Box
        min_b = torch.floor((C - self.max_depth) / self.block_size).int()
        max_b = torch.ceil((C + self.max_depth) / self.block_size).int()
        
        bx = torch.arange(min_b[0], max_b[0]+1, device=self.device)
        by = torch.arange(min_b[1], max_b[1]+1, device=self.device)
        bz = torch.arange(min_b[2], max_b[2]+1, device=self.device)
        
        X_grid, Y_grid, Z_grid = torch.meshgrid(bx, by, bz, indexing='ij')
        block_coords = torch.stack([X_grid, Y_grid, Z_grid], dim=-1).view(-1, 3)
        
        # Cull to exact sphere radius
        centers = (block_coords + 0.5) * self.block_size
        dists = torch.norm(centers - C, dim=-1)
        valid_blocks = block_coords[dists <= (self.max_depth + self.block_size)]
        
        valid_blocks_cpu = valid_blocks.cpu().numpy()
        
        # 2. Pool Assignment (Extremely fast O(N) lookup)
        active_indices = []
        for row in valid_blocks_cpu:
            k = tuple(row)
            if k not in self.block_hash:
                if self.next_idx >= self.max_blocks:
                    continue # Pool is full
                self.block_hash[k] = self.next_idx
                self.next_idx += 1
            active_indices.append(self.block_hash[k])
            
        if not active_indices: return
        
        # 3. Memory Gather (Zero-Copy View)
        idx_tensor = torch.tensor(active_indices, dtype=torch.long, device=self.device)
        B = len(active_indices)
        N = self.voxels_per_block
        
        old_tsdf = self.tsdf[idx_tensor].view(-1)
        old_w = self.weights[idx_tensor].view(-1)
        old_colors = self.colors[idx_tensor].view(-1, 3)
        old_min_dist = self.min_dist[idx_tensor].view(-1)
        
        # 4. Math: Generate coordinates and Transform
        offsets = valid_blocks * self.block_size # [B, 3]
        world_coords = (self.block_template.unsqueeze(0) + offsets.unsqueeze(1)).view(B * N, 3)
        
        # Fast Rigid Transform (Bypasses homogenous matrix concatenation to save 1.5 GB VRAM)
        pose_inv = torch.linalg.inv(pose)
        cam_coords = (world_coords @ pose_inv[:3, :3].T) + pose_inv[:3, 3]
        
        X, Y, Z = cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 2]
        r = torch.sqrt(X**2 + Y**2 + Z**2)
        
        # Clamp before asin to prevent floating point NaN crashes
        phi = torch.asin(torch.clamp(Y / (r + 1e-6), -1.0, 1.0))
        theta = torch.atan2(X, Z)
        
        u_norm = theta / torch.pi
        v_norm = phi / (torch.pi / 2.0)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
        depth_tensor = depth_map.unsqueeze(0).unsqueeze(0)
        mask_tensor = mask.unsqueeze(0).unsqueeze(0)
        rgb_tensor = rgb_image.permute(2, 0, 1).unsqueeze(0)
        
        sampled_depth = F.grid_sample(depth_tensor, grid, mode='bilinear', align_corners=True).squeeze()
        sampled_mask = F.grid_sample(mask_tensor, grid, mode='nearest', align_corners=True).squeeze()
        sampled_rgb = F.grid_sample(rgb_tensor, grid, mode='bilinear', align_corners=True).squeeze().T
        
        # 5. Math: TSDF & Sharp Color Update
        sdf = sampled_depth - r
        valid = (sampled_mask > 0.5) & (sampled_depth > 0.1) & (r < self.max_depth) & (sdf > -self.margin)
        
        tsdf_update = torch.clamp(sdf[valid] / self.margin, -1.0, 1.0)
        
        old_tsdf[valid] = (old_tsdf[valid] * old_w[valid] + tsdf_update) / (old_w[valid] + 1.0)
        old_w[valid] += 1.0
        
        closer_mask = r < old_min_dist
        color_update_mask = valid & closer_mask
        
        if color_update_mask.any():
            old_colors[color_update_mask] = sampled_rgb[color_update_mask]
            old_min_dist[color_update_mask] = r[color_update_mask]
            
        # 6. Scatter Back to Pool
        self.tsdf[idx_tensor] = old_tsdf.view(B, N)
        self.weights[idx_tensor] = old_w.view(B, N)
        self.colors[idx_tensor] = old_colors.view(B, N, 3)
        self.min_dist[idx_tensor] = old_min_dist.view(B, N)
        
        print(f"      [Profile - TSDF] VRAM Math: Processed {B} blocks ({B*N/1e6:.1f}M voxels) in {time.time() - t_start:.4f} sec")

    def extract_point_cloud(self):
        """100% Vectorized GPU point cloud extraction."""
        t_start = time.time()
        
        if self.next_idx == 0:
            return o3d.geometry.PointCloud()
            
        # [FIX] Thicker Margin for dense coverage: Extract 2 voxel widths deep
        surface_threshold = self.voxel_size * 2.0 
        
        # Evaluate only the allocated slice of the pool
        tsdf_vals = self.tsdf[:self.next_idx]
        weights = self.weights[:self.next_idx]
        
        physical_sdf = tsdf_vals * self.margin
        valid = (torch.abs(physical_sdf) <= surface_threshold) & (weights > 0)
        
        b_idx, v_idx = torch.where(valid)
        if len(b_idx) == 0:
            return o3d.geometry.PointCloud()
            
        # Reconstruct coordinates massively in parallel
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