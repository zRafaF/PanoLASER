import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from nvblox_torch.mapper import Mapper
from nvblox_torch.sensor import Sensor

class NvbloxPanoTSDF:
    def __init__(self, voxel_size_m=0.015, max_depth=6.0, face_size=512, device="cuda"):
        self.device = torch.device(device)
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth
        self.face_size = face_size
        
        print(f"[TSDF] Initializing dynamic GPU nvblox Mapper (Voxel Size: {voxel_size_m}m)...")
        self.mapper = Mapper(voxel_sizes_m=self.voxel_size_m)
        
        f = self.face_size / 2.0
        c = self.face_size / 2.0
        
        self.camera = Sensor.from_camera(
            fu=f, fv=f, cu=c, cv=c, 
            width=self.face_size, height=self.face_size
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
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
            pose = torch.from_numpy(pose).float().to(self.device)
            
        # 1. Prepare explicit PyTorch Mask Tensor
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device)
            else:
                mask = mask.float()
        else:
            mask = torch.ones_like(pano_depth_map)
            
        use_color = pano_rgb is not None
        if use_color and isinstance(pano_rgb, np.ndarray):
            pano_rgb = torch.from_numpy(pano_rgb).float().to(self.device)
            
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        pano_mask_tensor = mask.unsqueeze(0).unsqueeze(0)
        pano_rgb_tensor = pano_rgb.permute(2, 0, 1).unsqueeze(0) if use_color else None

        for i in range(6):
            # 2. NEAREST sampling prevents "ramp" artifacts at the mask edges
            radial_depth = F.grid_sample(
                pano_depth_tensor, 
                self.batched_grids[i:i+1], 
                mode='nearest', 
                align_corners=True
            ).squeeze(0).squeeze(0)
            
            face_mask = F.grid_sample(
                pano_mask_tensor,
                self.batched_grids[i:i+1],
                mode='nearest',
                align_corners=True
            ).squeeze(0).squeeze(0)
            
            optical_depth = radial_depth * self.batched_z_mults[i]
            
            # 3. STRICT INVALIDATION
            # nvblox ignores any depth < 0.0. This completely stops fake geometry.
            optical_depth[face_mask < 0.5] = -1.0
            optical_depth[optical_depth > self.max_depth] = -1.0
            optical_depth[optical_depth <= 0.01] = -1.0 
            
            color_face_uint8 = None
            if use_color:
                # Color can stay bilinear for visual smoothness
                color_face = F.grid_sample(
                    pano_rgb_tensor, 
                    self.batched_grids[i:i+1], 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                color_face_uint8 = color_face.permute(1, 2, 0).to(torch.uint8).contiguous()

            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            face_pose_cpu = face_pose.cpu()
            
            # Memory clearance before entering C++ backend
            del radial_depth, face_mask, face_pose
            if use_color:
                del color_face
            torch.cuda.empty_cache()
            
            self.mapper.add_depth_frame(
                optical_depth.contiguous(), 
                face_pose_cpu, 
                self.camera
            )
            
            if use_color:
                self.mapper.add_color_frame(
                    color_face_uint8,
                    face_pose_cpu,
                    self.camera
                )
                del color_face_uint8
                
            del optical_depth, face_pose_cpu

        # Final cleanup
        del pano_depth_tensor, pano_mask_tensor, pano_depth_map, pose, mask
        if use_color:
            del pano_rgb_tensor, pano_rgb
        torch.cuda.empty_cache()

    def extract_point_cloud(self, surface_threshold=0.02, viz_voxel_scale=4.0):
        self.mapper.update_color_mesh()
        o3d_mesh = self.mapper.get_color_mesh().to_open3d()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d_mesh.vertices
        pcd.colors = o3d_mesh.vertex_colors
        
        if viz_voxel_scale > 1.0:
            voxel_size = self.voxel_size_m * viz_voxel_scale
            pcd = pcd.voxel_down_sample(voxel_size)
            
        return pcd

    def extract_mesh(self):
        print("[TSDF] Generating dense planar mesh from nvblox...")
        self.mapper.update_color_mesh()
        return self.mapper.get_color_mesh().to_open3d()