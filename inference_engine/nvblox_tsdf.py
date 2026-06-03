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
        # Ensure depth is a tensor
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
        
        # Ensure pose is a tensor
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(self.device)
            
        # Ensure mask is a tensor
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.copy()).float().to(self.device)
            else:
                mask = mask.float()
        else:
            mask = torch.ones_like(pano_depth_map)
            
        # --- FIXED: Robust RGB Tensor Conversion ---
        use_color = pano_rgb is not None
        if use_color:
            if isinstance(pano_rgb, np.ndarray):
                pano_rgb = torch.from_numpy(pano_rgb).float().to(self.device)
            # Now it's guaranteed to be a tensor, so .permute() will work
            pano_rgb_tensor = pano_rgb.permute(2, 0, 1).unsqueeze(0)
        else:
            pano_rgb_tensor = None
            
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        pano_mask_tensor = mask.unsqueeze(0).unsqueeze(0)

        for i in range(6):
            try:
                # 2. Slice the map
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
                optical_depth[face_mask < 0.5] = -1.0
                optical_depth[optical_depth > self.max_depth] = -1.0
                
                face_pose = pose.clone()
                face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
                face_pose_cpu = face_pose.cpu()
                
                # 3. Integrate depth into nvblox
                self.mapper.add_depth_frame(
                    optical_depth.contiguous(),
                    face_pose_cpu,
                    self.camera
                )

                # 4. Integrate color into nvblox (this is what populates vertex_colors!)
                if pano_rgb_tensor is not None:
                    face_color = F.grid_sample(
                        pano_rgb_tensor,
                        self.batched_grids[i:i+1],
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)  # (3, H, W)
                    # nvblox expects (H, W, 3) uint8
                    face_color_hwc = face_color.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu()
                    self.mapper.add_color_frame(face_color_hwc, face_pose_cpu, self.camera)
                    del face_color, face_color_hwc

                # Cleanup local loop variables
                del radial_depth, optical_depth, face_pose, face_mask
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("[TSDF] !! CRITICAL OOM during integration. Skipping sub-frame...")
                    torch.cuda.empty_cache()
                else:
                    raise e

        # Final cleanup
        del pano_depth_tensor, pano_mask_tensor, pano_depth_map, pose
        torch.cuda.empty_cache()

    def extract_point_cloud(self, surface_threshold=0.02, viz_voxel_scale=4.0):
        # update_color_mesh called ONCE here; extract_mesh reuses the result
        # without calling it again, so pcd.colors won't be invalidated.
        self.mapper.update_color_mesh()
        o3d_mesh = self.mapper.get_color_mesh().to_open3d()

        pcd = o3d.geometry.PointCloud()
        # Force numpy copies so pcd owns its data independently of the mesh buffer
        pcd.points = o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertices).copy())
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertex_colors).copy())

        if viz_voxel_scale > 1.0:
            voxel_size = self.voxel_size_m * viz_voxel_scale
            pcd = pcd.voxel_down_sample(voxel_size)

        return pcd

    def extract_mesh(self):
        print("[TSDF] Generating dense planar mesh from nvblox...")
        # Reuse the mesh already updated by extract_point_cloud —
        # calling update_color_mesh() again would invalidate the pcd color buffer.
        return self.mapper.get_color_mesh().to_open3d()