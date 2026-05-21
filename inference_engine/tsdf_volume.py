import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time

class BlockSparseSphericalTSDF:
    def __init__(self, voxel_size=0.02, margin=0.08, block_res=32, max_depth=4.0, device="cuda"):
        """
        An infinitely scalable, GPU-native Block-Sparse TSDF.
        block_res: How many voxels per side in a memory block (32^3 = 32,768 voxels per block).
        max_depth: How far the camera can see (meters). Limits the active block sphere.
        """
        self.device = device
        self.voxel_size = voxel_size
        self.margin = margin
        self.max_depth = max_depth
        
        self.block_res = block_res
        self.block_size = block_res * voxel_size
        self.voxels_per_block = block_res ** 3
        
        # Hash map of active space: (x, y, z) tuple -> Dict of tensors
        self.blocks = {}
        
        # Pre-compute the local 3D coordinates for a generic block (saves VRAM)
        x = torch.arange(self.block_res, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing='ij')
        self.block_template = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3).float() * self.voxel_size

        print(f"[TSDF] Initialized Infinite Block-Sparse Grid. Voxel: {voxel_size}m, Block: {self.block_size:.2f}m")

    def _allocate_block(self, key):
        """Allocates a new 32x32x32 voxel block in VRAM."""
        self.blocks[key] = {
            'tsdf': torch.ones(self.voxels_per_block, dtype=torch.float32, device=self.device),
            'weights': torch.zeros(self.voxels_per_block, dtype=torch.float32, device=self.device),
            'colors': torch.zeros((self.voxels_per_block, 3), dtype=torch.float32, device=self.device)
        }

    @torch.no_grad()
    def integrate(self, depth_map, rgb_image, mask, pose):
        t_start = time.time()
        
        # Ensure inputs are tensors on the right device
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map).float().to(self.device)
            mask = torch.from_numpy(mask).float().to(self.device)
            rgb_image = torch.from_numpy(rgb_image).float().to(self.device) / 255.0
            pose = torch.from_numpy(pose).float().to(self.device)
            
        C = pose[:3, 3] # Camera position
        
        # 1. Find all Blocks within the camera's visual radius (max_depth)
        min_b = torch.floor((C - self.max_depth) / self.block_size).int().cpu().numpy()
        max_b = torch.ceil((C + self.max_depth) / self.block_size).int().cpu().numpy()
        
        active_keys = []
        for x in range(min_b[0], max_b[0] + 1):
            for y in range(min_b[1], max_b[1] + 1):
                for z in range(min_b[2], max_b[2] + 1):
                    # Prune corners of the bounding box to form a sphere
                    center = torch.tensor([x + 0.5, y + 0.5, z + 0.5], device=self.device) * self.block_size
                    if torch.norm(center - C) <= (self.max_depth + self.block_size):
                        key = (x, y, z)
                        active_keys.append(key)
                        if key not in self.blocks:
                            self._allocate_block(key)
                            
        if not active_keys: return
        
        # 2. Gather active blocks into a single batched tensor for lightning-fast math
        B = len(active_keys)
        N = self.voxels_per_block
        
        old_tsdf = torch.stack([self.blocks[k]['tsdf'] for k in active_keys]).view(B * N)
        old_w = torch.stack([self.blocks[k]['weights'] for k in active_keys]).view(B * N)
        old_colors = torch.stack([self.blocks[k]['colors'] for k in active_keys]).view(B * N, 3)
        
        # Calculate true world coordinates for these active blocks on the fly
        offsets = torch.tensor(active_keys, device=self.device) * self.block_size
        world_coords = (self.block_template.unsqueeze(0) + offsets.unsqueeze(1)).view(B * N, 3)
        
        # 3. Spherical Projection (Exactly as before, but bounded to active memory)
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
        
        # 4. TSDF Update logic
        sdf = sampled_depth - r
        
        # Valid if: Masked, Depth > 0, Ray is within max_depth, and SDF > -margin
        valid = (sampled_mask > 0.5) & (sampled_depth > 0.1) & (r < self.max_depth) & (sdf > -self.margin)
        
        tsdf_update = torch.clamp(sdf[valid] / self.margin, -1.0, 1.0)
        
        new_w = 1.0 
        old_tsdf[valid] = (old_tsdf[valid] * old_w[valid] + tsdf_update * new_w) / (old_w[valid] + new_w)
        old_colors[valid] = (old_colors[valid] * old_w[valid].unsqueeze(1) + sampled_rgb[valid] * new_w) / (old_w[valid].unsqueeze(1) + new_w)
        old_w[valid] += new_w
        
        # 5. Scatter the updated data back into the discrete block dictionary
        new_tsdf = old_tsdf.view(B, N)
        new_weights = old_w.view(B, N)
        new_colors = old_colors.view(B, N, 3)
        
        for i, k in enumerate(active_keys):
            self.blocks[k]['tsdf'] = new_tsdf[i]
            self.blocks[k]['weights'] = new_weights[i]
            self.blocks[k]['colors'] = new_colors[i]
            
        print(f"      [Profile - TSDF] Processed {B} blocks ({B*N} voxels) in {time.time() - t_start:.4f} sec")

    def extract_point_cloud(self, surface_threshold=0.02):
        """Extracts exactly the surfaces from all allocated blocks."""
        t_start = time.time()
        all_points, all_colors = [], []
        
        for k, block in self.blocks.items():
            valid = (torch.abs(block['tsdf']) < surface_threshold) & (block['weights'] > 0)
            if not valid.any(): continue
            
            offset = torch.tensor(k, device=self.device) * self.block_size
            coords = self.block_template + offset
            
            all_points.append(coords[valid])
            all_colors.append(block['colors'][valid])
            
        if not all_points:
            return o3d.geometry.PointCloud()
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(torch.cat(all_points).cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(torch.cat(all_colors).cpu().numpy())
        
        print(f"      [Profile - Extraction] Extracted {len(pcd.points)} points in {time.time() - t_start:.4f} sec")
        return pcd