import torch
import torch.nn.functional as F
import open3d as o3d
import numpy as np

class Open3DPanoVBG:
    def __init__(self, voxel_size_m=0.01, max_depth=6.0, face_size=512, device="cuda"):
        # Keep PyTorch on the GPU for fast slicing
        self.torch_device = torch.device(device)
        
        # STRATEGIC PIVOT: Force Open3D entirely onto the CPU/System RAM
        print("[TSDF] Offloading Open3D VoxelBlockGrid to CPU RAM to preserve VRAM...")
        self.o3d_device = o3d.core.Device("CPU:0") 
        self.cpu_device = o3d.core.Device("CPU:0") 
        
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth
        self.face_size = face_size
        
        # Because we are using System RAM, we can afford a massive block pool
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=['tsdf', 'weight', 'color'],
            attr_dtypes=[o3d.core.float32, o3d.core.float32, o3d.core.float32],
            attr_channels=[[1], [1], [3]],
            voxel_size=self.voxel_size_m,
            block_resolution=16,
            block_count=50000, # Massive map capacity, 0 bytes of VRAM used
            device=self.o3d_device
        )
        
        f = self.face_size / 2.0
        c = self.face_size / 2.0
        self.intrinsic_np = np.array([
            [f, 0, c],
            [0, f, c],
            [0, 0, 1]
        ], dtype=np.float64)
        self.intrinsic_o3d = o3d.core.Tensor(self.intrinsic_np, o3d.core.float64, self.cpu_device)
        
        self._precompute_cubemap_grids()

    def _precompute_cubemap_grids(self):
        u = torch.linspace(-1, 1, self.face_size, device=self.torch_device)
        v = torch.linspace(-1, 1, self.face_size, device=self.torch_device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        face_dirs = [
            torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1),   # Front
            torch.stack([torch.ones_like(u_grid), v_grid, -u_grid], dim=-1),  # Right
            torch.stack([-u_grid, v_grid, -torch.ones_like(u_grid)], dim=-1), # Back
            torch.stack([-torch.ones_like(u_grid), v_grid, u_grid], dim=-1),  # Left
            torch.stack([u_grid, -torch.ones_like(u_grid), v_grid], dim=-1),  # Top
            torch.stack([u_grid, torch.ones_like(u_grid), -v_grid], dim=-1)   # Bottom
        ]
        
        self.face_rotations = [
            torch.eye(3, device=self.torch_device), 
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.torch_device, dtype=torch.float32), 
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.torch_device, dtype=torch.float32), 
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.torch_device, dtype=torch.float32), 
            torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=self.torch_device, dtype=torch.float32), 
            torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.torch_device, dtype=torch.float32), 
        ]
        
        grids = []
        for d in face_dirs:
            d_norm = F.normalize(d, p=2, dim=-1)
            X, Y, Z = d_norm[..., 0], d_norm[..., 1], d_norm[..., 2]
            
            theta = torch.atan2(X, Z) 
            phi = torch.asin(torch.clamp(Y, -1.0, 1.0)) 
            
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            grids.append(torch.stack([u_norm, v_norm], dim=-1))
            
        z_mult = 1.0 / torch.sqrt(u_grid**2 + v_grid**2 + 1.0)
        self.batched_z_mults = z_mult.unsqueeze(0).expand(6, -1, -1)
        self.batched_grids = torch.stack(grids, dim=0)

    @torch.no_grad()
    def integrate(self, pano_depth_map, pano_rgb, mask, pose):
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.torch_device)
            pose = torch.from_numpy(pose).float().to(self.torch_device)
            
        use_color = pano_rgb is not None
        if use_color and isinstance(pano_rgb, np.ndarray):
            pano_rgb = torch.from_numpy(pano_rgb).float().to(self.torch_device)

        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        pano_rgb_tensor = pano_rgb.permute(2, 0, 1).unsqueeze(0) if use_color else None

        for i in range(6):
            # 1. Slice extremely fast on the GPU
            radial_depth = F.grid_sample(
                pano_depth_tensor, 
                self.batched_grids[i:i+1], 
                mode='bilinear', 
                align_corners=True
            ).squeeze(0).squeeze(0)
            
            optical_depth = radial_depth * self.batched_z_mults[i]
            optical_depth[optical_depth > self.max_depth] = 0.0
            
            color_face = None
            if use_color:
                color_face = F.grid_sample(
                    pano_rgb_tensor, 
                    self.batched_grids[i:i+1], 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                color_face = color_face.permute(1, 2, 0) / 255.0

            # 2. Map Pose to CPU
            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            extrinsic_np = torch.linalg.inv(face_pose).cpu().numpy()
            extrinsic_o3d = o3d.core.Tensor(extrinsic_np, o3d.core.float64, self.cpu_device)
            
            # 3. Pull Tensors to CPU System RAM via numpy
            depth_np = optical_depth.cpu().numpy()
            depth_o3d = o3d.t.geometry.Image(o3d.core.Tensor(depth_np, o3d.core.float32, self.cpu_device))
            
            color_o3d = None
            if use_color:
                color_np = color_face.cpu().numpy()
                color_o3d = o3d.t.geometry.Image(o3d.core.Tensor(color_np, o3d.core.float32, self.cpu_device))
                
            block_coords = self.vbg.compute_unique_block_coordinates(
                depth_o3d, self.intrinsic_o3d, extrinsic_o3d, depth_scale=1.0, depth_max=self.max_depth
            )

            # 4. Immediate GPU Memory Release
            del radial_depth, optical_depth, face_pose
            if use_color:
                del color_face
                        
            # 5. Integrate cleanly on the CPU (Zero VRAM impact)
            self.vbg.integrate(
                block_coords=block_coords,
                depth=depth_o3d,
                color=color_o3d,
                depth_intrinsic=self.intrinsic_o3d,
                color_intrinsic=self.intrinsic_o3d,
                extrinsic=extrinsic_o3d,
                depth_scale=1.0,
                depth_max=self.max_depth
            )
            
        del pano_depth_tensor, pano_depth_map, pose
        if use_color:
            del pano_rgb_tensor, pano_rgb

    def extract_point_cloud(self):
        # Extraction happens on CPU, seamlessly passed to the Gradio server
        return self.vbg.extract_point_cloud(weight_threshold=3.0).to_legacy()

    def extract_mesh(self):
        print("[TSDF] Extracting high-density mesh on CPU...")
        t_mesh = self.vbg.extract_triangle_mesh(weight_threshold=3.0)
        return t_mesh.to_legacy()