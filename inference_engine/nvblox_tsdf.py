import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from nvblox_torch.mapper import Mapper
from nvblox_torch.sensor import Sensor

class NvbloxPanoTSDF:
    def __init__(self, voxel_size_m=0.01, max_depth=3.5, face_size=512, crop_margin=24, device="cuda"):
        self.device = torch.device(device)
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth 
        self.face_size = face_size
        self.crop_margin = crop_margin
        
        print(f"[TSDF] Initializing 1cm Global GPU Mapper (Forcefield & {max_depth}m Distance Cap Active)...")
        self.mapper = Mapper(voxel_sizes_m=self.voxel_size_m)
        
        f = self.face_size / 2.0
        w = self.face_size - (2 * self.crop_margin)
        h = self.face_size - (2 * self.crop_margin)
        c = w / 2.0 
        
        self.camera = Sensor.from_camera(
            fu=f, fv=f, cu=c, cv=c, 
            width=w, height=h
        )
        
        self._precompute_cubemap_grids()

    def _precompute_cubemap_grids(self):
        u = torch.linspace(-1, 1, self.face_size, device=self.device)
        v = torch.linspace(-1, 1, self.face_size, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        base_rays = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
        
        self.face_rotations = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device, dtype=torch.float32), 
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.device, dtype=torch.float32),
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32),
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32),
            torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], device=self.device, dtype=torch.float32),
            torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], device=self.device, dtype=torch.float32),
        ]
        
        grids = []
        for R in self.face_rotations:
            ray_global = base_rays @ R.T
            X, Y, Z = ray_global[..., 0], ray_global[..., 1], ray_global[..., 2]
            norm = torch.sqrt(X**2 + Y**2 + Z**2)
            theta = torch.atan2(X, Z) 
            phi = torch.asin(torch.clamp(Y / norm, -1.0, 1.0)) 
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            grids.append(torch.stack([u_norm, v_norm], dim=-1))
            
        self.batched_grids = torch.stack(grids, dim=0)
        z_mult = 1.0 / torch.sqrt(u_grid**2 + v_grid**2 + 1.0)
        self.batched_z_mults = z_mult.unsqueeze(0).expand(6, -1, -1)

    @torch.no_grad()
    def integrate(self, pano_depth_map, pano_rgb, mask, pose):
        # Move inputs to device seamlessly
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(self.device)
            
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.copy()).float().to(self.device)
            else:
                mask = mask.float()
        else:
            mask = torch.ones_like(pano_depth_map)
            
        # =========================================================
        # OPTIMIZED PROTECTIVE FORCEFIELD 
        # =========================================================
        # Using torch.roll to avoid intermediate memory allocations
        depth_shifted_x = torch.roll(pano_depth_map, shifts=-1, dims=1)
        depth_shifted_y = torch.roll(pano_depth_map, shifts=-1, dims=0)
        
        diff_x = torch.abs(pano_depth_map - depth_shifted_x)
        diff_y = torch.abs(pano_depth_map - depth_shifted_y)
        
        # 1. 8cm Imprinting Mask (Slices away local fuzz)
        edge_mask = (diff_x < 0.08) & (diff_y < 0.08)
        mask = mask * edge_mask.float()
        
        # 2. Thin-Object Forcefield (Prevents far rays from erasing door frames)
        silhouette_edges = ((diff_x > 0.20) | (diff_y > 0.20)).float()
        edges_tensor = silhouette_edges.unsqueeze(0).unsqueeze(0)
        dilated_edges = F.max_pool2d(edges_tensor, kernel_size=5, stride=1, padding=2).squeeze()
        
        # Apply forcefield blind spot
        mask = mask * (dilated_edges == 0.0).float()
        # =========================================================

        use_color = pano_rgb is not None
        if use_color:
            if isinstance(pano_rgb, np.ndarray):
                pano_rgb = torch.from_numpy(pano_rgb).float().to(self.device)
            pano_rgb_tensor = pano_rgb.permute(2, 0, 1).unsqueeze(0)
        else:
            pano_rgb_tensor = None
            
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        pano_mask_tensor = mask.unsqueeze(0).unsqueeze(0)

        for i in range(6):
            radial_depth = F.grid_sample(
                pano_depth_tensor, self.batched_grids[i:i+1], mode='nearest', align_corners=True
            ).squeeze(0).squeeze(0)
            
            face_mask = F.grid_sample(
                pano_mask_tensor, self.batched_grids[i:i+1], mode='nearest', align_corners=True
            ).squeeze(0).squeeze(0)
            
            # Bottom face circular tripod mask
            if i == 5:
                u_coords = self.batched_grids[i, :, :, 0]
                v_coords = self.batched_grids[i, :, :, 1]
                radius_sq = u_coords**2 + v_coords**2
                nadir_mask = (radius_sq > 0.35**2).float() 
                face_mask = face_mask * nadir_mask

            optical_depth = radial_depth * self.batched_z_mults[i]
            optical_depth[face_mask < 0.5] = -1.0
            
            # Cap the integration distance.
            optical_depth[optical_depth > self.max_depth] = -1.0
            
            color_face_uint8 = None
            if use_color and i != 5: # Skip coloring the floor/tripod nadir 
                color_face = F.grid_sample(
                    pano_rgb_tensor, self.batched_grids[i:i+1], mode='bilinear', align_corners=True
                ).squeeze(0)
                
                # Convert safely to uint8 for Nvblox mapping
                if color_face.is_floating_point() and color_face.max() <= 1.0:
                    color_face = color_face * 255.0
                color_face_uint8 = color_face.permute(1, 2, 0).to(torch.uint8).contiguous()

            if self.crop_margin > 0:
                c = self.crop_margin
                optical_depth = optical_depth[c:-c, c:-c].contiguous()
                if color_face_uint8 is not None:
                    color_face_uint8 = color_face_uint8[c:-c, c:-c, :].contiguous()

            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            face_pose_cpu = face_pose.cpu()
            
            # Integrate into the global TSDF
            self.mapper.add_depth_frame(optical_depth, face_pose_cpu, self.camera)
            
            if color_face_uint8 is not None:
                self.mapper.add_color_frame(color_face_uint8, face_pose_cpu, self.camera)

        # Allow PyTorch to automatically manage its cache, no forced empties
        del pano_depth_tensor, pano_mask_tensor

    def extract_mesh(self):
        """Extracts the global fused mesh. Call this only when necessary."""
        self.mapper.update_color_mesh()
        return self.mapper.get_color_mesh().to_open3d()