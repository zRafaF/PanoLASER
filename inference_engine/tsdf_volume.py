import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time

class SphericalTSDFVolume:
    def __init__(self, vol_bounds, voxel_size=0.02, margin=0.1, device="cuda"):
        """
        Initializes a GPU-native dense TSDF volume.
        vol_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        self.device = device
        self.voxel_size = voxel_size
        self.margin = margin
        
        self.bounds = torch.tensor(vol_bounds, dtype=torch.float32, device=self.device)
        self.vol_dim = torch.ceil((self.bounds[:, 1] - self.bounds[:, 0]) / self.voxel_size).long()
        
        # 1. Generate dense voxel grid coordinates
        x = torch.arange(self.vol_dim[0], dtype=torch.float32, device=self.device)
        y = torch.arange(self.vol_dim[1], dtype=torch.float32, device=self.device)
        z = torch.arange(self.vol_dim[2], dtype=torch.float32, device=self.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        self.vox_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
        self.world_coords = self.bounds[:, 0] + self.vox_coords * self.voxel_size
        
        # 2. Initialize TSDF and Weights
        self.num_voxels = self.world_coords.shape[0]
        self.tsdf = torch.ones(self.num_voxels, dtype=torch.float32, device=self.device)
        self.weights = torch.zeros(self.num_voxels, dtype=torch.float32, device=self.device)
        
        # We also want to colorize it. Initialize an RGB volume.
        self.colors = torch.zeros((self.num_voxels, 3), dtype=torch.float32, device=self.device)

        print(f"[TSDF] Initialized {self.vol_dim[0]}x{self.vol_dim[1]}x{self.vol_dim[2]} grid ({self.num_voxels} voxels) on {self.device}.")

    @torch.no_grad()
    def integrate(self, depth_map, rgb_image, mask, pose, scale=1.0):
        """
        Projects the volume into the equirectangular camera to update SDF.
        """
        t_start = time.time()
        
        # Ensure inputs are tensors on the right device
        if isinstance(depth_map, np.ndarray):
            depth_map = torch.from_numpy(depth_map).float().to(self.device)
            mask = torch.from_numpy(mask).float().to(self.device)
            rgb_image = torch.from_numpy(rgb_image).float().to(self.device) / 255.0
            pose = torch.from_numpy(pose).float().to(self.device)
            
        H, W = depth_map.shape
        
        # 1. Transform World Voxels to Camera Space
        # world_coords_homo: [N, 4]
        world_homo = torch.cat([self.world_coords, torch.ones((self.num_voxels, 1), device=self.device)], dim=1)
        pose_inv = torch.linalg.inv(pose)
        
        # cam_coords: [N, 3]
        cam_coords = (world_homo @ pose_inv.T)[:, :3]
        
        # 2. Spherical Projection (Cartesian to Theta/Phi)
        X, Y, Z = cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 2]
        
        # Voxel distance from camera
        r = torch.sqrt(X**2 + Y**2 + Z**2)
        
        # Spherical angles
        phi = torch.asin(Y / r)         # Latitude: [-pi/2, pi/2]
        theta = torch.atan2(X, Z)       # Longitude: [-pi, pi]
        
        # Normalize to [-1, 1] for grid_sample
        u_norm = theta / torch.pi
        v_norm = phi / (torch.pi / 2.0)
        
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0) # Shape: [1, 1, N, 2]
        
        # 3. Sample Depth and Masks
        # We need to reshape maps for grid_sample: [B, C, H, W]
        depth_tensor = depth_map.unsqueeze(0).unsqueeze(0) * scale 
        mask_tensor = mask.unsqueeze(0).unsqueeze(0)
        rgb_tensor = rgb_image.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        
        # grid_sample uses bilinear interpolation instantly on millions of points
        sampled_depth = F.grid_sample(depth_tensor, grid, mode='bilinear', align_corners=True).squeeze()
        sampled_mask = F.grid_sample(mask_tensor, grid, mode='nearest', align_corners=True).squeeze()
        sampled_rgb = F.grid_sample(rgb_tensor, grid, mode='bilinear', align_corners=True).squeeze().T # [N, 3]
        
        # 4. Calculate Signed Distance
        sdf = sampled_depth - r
        
        # 5. Determine Valid Integration Zone
        # Valid if: Ray hit the mask, Depth is positive, and voxel is NOT way behind the surface
        valid = (sampled_mask > 0.5) & (sampled_depth > 0.1) & (sdf > -self.margin)
        
        # 6. Truncate and Update
        tsdf_update = torch.clamp(sdf[valid] / self.margin, -1.0, 1.0)
        
        old_tsdf = self.tsdf[valid]
        old_w = self.weights[valid]
        
        new_w = 1.0 # Simple constant weighting
        
        # Running average formula
        self.tsdf[valid] = (old_tsdf * old_w + tsdf_update * new_w) / (old_w + new_w)
        self.colors[valid] = (self.colors[valid] * old_w.unsqueeze(1) + sampled_rgb[valid] * new_w) / (old_w.unsqueeze(1) + new_w)
        self.weights[valid] += new_w
        
        print(f"      [Profile - TSDF] Raycast integration took {time.time() - t_start:.4f} sec")

    def extract_point_cloud(self, surface_threshold=0.15):
        """
        Extracts active surface voxels into an Open3D point cloud.
        """
        t_start = time.time()
        
        # Surface voxels are near zero-crossing and have been observed
        surface_mask = (torch.abs(self.tsdf) < surface_threshold) & (self.weights > 0)
        
        surface_coords = self.world_coords[surface_mask].cpu().numpy()
        surface_colors = self.colors[surface_mask].cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_coords)
        pcd.colors = o3d.utility.Vector3dVector(surface_colors)
        
        print(f"      [Profile - Extraction] Extracted {len(pcd.points)} points in {time.time() - t_start:.4f} sec")
        return pcd