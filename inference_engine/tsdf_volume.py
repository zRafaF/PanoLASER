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
        
        centers = (block_coords + 0.5) * self.block_size
        dists = torch.norm(centers - C, dim=-1)
        valid_blocks = block_coords[dists <= (self.max_depth + self.block_size)]
        
        valid_blocks_cpu = valid_blocks.cpu().numpy()
        
        # 2. Pool Assignment (Fixing the PyTorch List Warning!)
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
            kept_indices.append(i)  # Keep the index, not the array
            
        if not active_indices: return
        
        # Slice the tensor directly, bypassing list conversion completely!
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
            valid = (sampled_mask > 0.5) & (sampled_depth > 0.1) & (r < self.max_depth) & (sdf > -self.margin)
            
            tsdf_update = torch.clamp(sdf[valid] / self.margin, -1.0, 1.0)
            
            old_tsdf[valid] = (old_tsdf[valid] * old_w[valid] + tsdf_update) / (old_w[valid] + 1.0)
            old_w[valid] += 1.0
            
            closer_mask = r < old_min_dist
            color_update_mask = valid & closer_mask
            
            if color_update_mask.any():
                old_colors[color_update_mask] = sampled_rgb[color_update_mask]
                old_min_dist[color_update_mask] = r[color_update_mask]
                
            self.tsdf[idx_tensor] = old_tsdf.view(B, N)
            self.weights[idx_tensor] = old_w.view(B, N)
            self.colors[idx_tensor] = old_colors.view(B, N, 3)
            self.min_dist[idx_tensor] = old_min_dist.view(B, N)
            
        # Optional sync inside TSDF class to get exact integration time per frame
        # torch.cuda.synchronize()
        # print(f"      [Profile - TSDF] VRAM Math: Processed {B_total} blocks ({B_total*self.voxels_per_block/1e6:.1f}M voxels) in {time.time() - t_start:.4f} sec")

    # [FIX] Added surface_threshold argument to prevent TypeError
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