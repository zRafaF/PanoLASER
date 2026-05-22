import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time

class BlockSparseSphericalTSDF:
    def __init__(self, voxel_size=0.02, margin=0.08, block_res=32, max_depth=4.0, device="cuda"):
        """
        An infinitely scalable Block-Sparse TSDF with CPU/GPU Paging and Sharp Color Overwrite.
        """
        self.device = device
        self.voxel_size = voxel_size
        self.margin = margin
        self.max_depth = max_depth
        
        self.block_res = block_res
        self.block_size = block_res * voxel_size
        self.voxels_per_block = block_res ** 3
        
        self.blocks = {}
        
        # Pre-compute the local 3D coordinates template on CPU to save VRAM
        x = torch.arange(self.block_res, device='cpu')
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        self.block_template = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).float() * self.voxel_size

        print(f"[TSDF] Initialized Paged Sparse Grid. Voxel: {voxel_size}m, Block: {self.block_size:.2f}m")

    def _allocate_block(self, key):
        """Allocates a new block purely on CPU RAM initially."""
        self.blocks[key] = {
            'tsdf': torch.ones(self.voxels_per_block, dtype=torch.float32, device='cpu'),
            'weights': torch.zeros(self.voxels_per_block, dtype=torch.float32, device='cpu'),
            'colors': torch.zeros((self.voxels_per_block, 3), dtype=torch.float32, device='cpu'),
            'min_dist': torch.full((self.voxels_per_block,), float('inf'), dtype=torch.float32, device='cpu'), # Tracks closest camera
            'on_gpu': False
        }

    def _page_memory(self, active_keys):
        """Moves active blocks to VRAM, and ships inactive blocks back to CPU RAM."""
        active_set = set(active_keys)
        
        for key, block in self.blocks.items():
            if key in active_set and not block['on_gpu']:
                # Page IN to GPU
                block['tsdf'] = block['tsdf'].to(self.device, non_blocking=True)
                block['weights'] = block['weights'].to(self.device, non_blocking=True)
                block['colors'] = block['colors'].to(self.device, non_blocking=True)
                block['min_dist'] = block['min_dist'].to(self.device, non_blocking=True)
                block['on_gpu'] = True
                
            elif key not in active_set and block['on_gpu']:
                # Page OUT to CPU
                block['tsdf'] = block['tsdf'].to('cpu', non_blocking=True)
                block['weights'] = block['weights'].to('cpu', non_blocking=True)
                block['colors'] = block['colors'].to('cpu', non_blocking=True)
                block['min_dist'] = block['min_dist'].to('cpu', non_blocking=True)
                block['on_gpu'] = False

    @torch.no_grad()
    def integrate(self, depth_map, rgb_image, mask, pose):
        t_start = time.time()
        
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map.copy()).float().to(self.device)
            mask = torch.from_numpy(mask.copy()).float().to(self.device)
            rgb_image = torch.from_numpy(rgb_image.copy()).float().to(self.device) / 255.0
            pose = torch.from_numpy(pose.copy()).float().to(self.device)
            
        C = pose[:3, 3]
        
        # 1. Identify active frustum
        min_b = torch.floor((C - self.max_depth) / self.block_size).int().cpu().numpy()
        max_b = torch.ceil((C + self.max_depth) / self.block_size).int().cpu().numpy()
        
        active_keys = []
        for x in range(min_b[0], max_b[0] + 1):
            for y in range(min_b[1], max_b[1] + 1):
                for z in range(min_b[2], max_b[2] + 1):
                    center = torch.tensor([x + 0.5, y + 0.5, z + 0.5], device=self.device) * self.block_size
                    if torch.norm(center - C) <= (self.max_depth + self.block_size):
                        key = (x, y, z)
                        if key not in self.blocks:
                            self._allocate_block(key)
                        active_keys.append(key)
                        
        if not active_keys: return
        
        # 2. Page memory (Maintains constant VRAM usage!)
        self._page_memory(active_keys)
        
        B = len(active_keys)
        N = self.voxels_per_block
        
        old_tsdf = torch.stack([self.blocks[k]['tsdf'] for k in active_keys]).view(B * N)
        old_w = torch.stack([self.blocks[k]['weights'] for k in active_keys]).view(B * N)
        old_colors = torch.stack([self.blocks[k]['colors'] for k in active_keys]).view(B * N, 3)
        old_min_dist = torch.stack([self.blocks[k]['min_dist'] for k in active_keys]).view(B * N)
        
        offsets = torch.tensor(active_keys, device=self.device) * self.block_size
        world_coords = (self.block_template.to(self.device).unsqueeze(0) + offsets.unsqueeze(1)).view(B * N, 3)
        
        # 3. Spherical Projection
        world_homo = torch.cat([world_coords, torch.ones((B * N, 1), device=self.device)], dim=1)
        pose_inv = torch.linalg.inv(pose)
        cam_coords = (world_homo @ pose_inv.T)[:, :3]
        
        X, Y, Z = cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 2]
        r = torch.sqrt(X**2 + Y**2 + Z**2)
        
        phi = torch.asin(Y / (r + 1e-6))
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
        
        # 4. TSDF & Sharp Color Update
        sdf = sampled_depth - r
        valid = (sampled_mask > 0.5) & (sampled_depth > 0.1) & (r < self.max_depth) & (sdf > -self.margin)
        
        tsdf_update = torch.clamp(sdf[valid] / self.margin, -1.0, 1.0)
        
        # Geometry Update (Running Average)
        old_tsdf[valid] = (old_tsdf[valid] * old_w[valid] + tsdf_update) / (old_w[valid] + 1.0)
        old_w[valid] += 1.0
        
        # SHARP COLOR OVERWRITE: Only update color if this observation is physically closer
        closer_mask = r < old_min_dist
        color_update_mask = valid & closer_mask
        
        if color_update_mask.any():
            old_colors[color_update_mask] = sampled_rgb[color_update_mask]
            old_min_dist[color_update_mask] = r[color_update_mask]
        
        # 5. Scatter back to memory blocks
        new_tsdf = old_tsdf.view(B, N)
        new_weights = old_w.view(B, N)
        new_colors = old_colors.view(B, N, 3)
        new_min_dist = old_min_dist.view(B, N)
        
        for i, k in enumerate(active_keys):
            self.blocks[k]['tsdf'] = new_tsdf[i]
            self.blocks[k]['weights'] = new_weights[i]
            self.blocks[k]['colors'] = new_colors[i]
            self.blocks[k]['min_dist'] = new_min_dist[i]
            
        print(f"      [Profile - TSDF] Paged & Processed {B} blocks ({B*N} voxels) in {time.time() - t_start:.4f} sec")

    def extract_point_cloud(self, surface_threshold=None):
        """Extracts surfaces cleanly using CPU RAM to prevent extraction crashes."""
        t_start = time.time()
        all_points, all_colors = [], []
        
        if surface_threshold is None:
            surface_threshold = self.voxel_size
            
        for k, block in self.blocks.items():
            # Perform extraction natively on whatever device the block currently lives on!
            dev = 'cuda' if block['on_gpu'] else 'cpu'
            
            physical_sdf = block['tsdf'] * self.margin
            valid = (torch.abs(physical_sdf) <= surface_threshold) & (block['weights'] > 0)
            
            if not valid.any(): continue
            
            offset = torch.tensor(k, device=dev) * self.block_size
            coords = self.block_template.to(dev) + offset
            
            all_points.append(coords[valid].cpu())
            all_colors.append(block['colors'][valid].cpu())
            
        if not all_points:
            return o3d.geometry.PointCloud()
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(torch.cat(all_points).numpy())
        pcd.colors = o3d.utility.Vector3dVector(torch.cat(all_colors).numpy())
        
        print(f"      [Profile - Extraction] Extracted {len(pcd.points)} sharp points in {time.time() - t_start:.4f} sec")
        return pcd