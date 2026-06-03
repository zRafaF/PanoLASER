import torch
import torch.nn.functional as F
import torch.utils.dlpack
import open3d as o3d
import numpy as np

class Open3DPanoVBG:
    def __init__(self, voxel_size_m=0.01, max_depth=6.0, face_size=512, device="cuda"):
        self.torch_device = torch.device(device)
        
        o3d_dev_str = device.upper()
        if ":" not in o3d_dev_str:
            o3d_dev_str += ":0"
            
        self.o3d_device = o3d.core.Device(o3d_dev_str)
        self.cpu_device = o3d.core.Device("CPU:0") # Explicit CPU device for matrices
        
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth
        self.face_size = face_size
        
        print(f"[TSDF] Initializing GPU Open3D VoxelBlockGrid (Voxel Size: {voxel_size_m}m)...")
        
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=['tsdf', 'weight', 'color'],
            attr_dtypes=[o3d.core.float32, o3d.core.float32, o3d.core.float32],
            attr_channels=[[1], [1], [3]],
            voxel_size=self.voxel_size_m,
            block_resolution=16,
            block_count=25000, 
            device=self.o3d_device
        )
        
        # Intrinsics mapped strictly to CPU
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
        """Precomputes and stacks the PyTorch grids for vectorized 6-face sampling."""
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
            torch.eye(3, device=self.torch_device), # Front
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.torch_device, dtype=torch.float32), # Right
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.torch_device, dtype=torch.float32), # Back
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.torch_device, dtype=torch.float32), # Left
            torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=self.torch_device, dtype=torch.float32), # Top
            torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.torch_device, dtype=torch.float32), # Bottom
        ]
        
        grids = []
        z_mults = []
        for d in face_dirs:
            d_norm = F.normalize(d, p=2, dim=-1)
            X, Y, Z = d_norm[..., 0], d_norm[..., 1], d_norm[..., 2]
            
            theta = torch.atan2(X, Z) 
            phi = torch.asin(torch.clamp(Y, -1.0, 1.0)) 
            
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            grids.append(torch.stack([u_norm, v_norm], dim=-1))
            z_mults.append(Z.abs())
            
        # Stack into single tensors for 1-shot batch processing [6, H, W, ...]
        self.batched_grids = torch.stack(grids, dim=0)
        self.batched_z_mults = torch.stack(z_mults, dim=0)

    @torch.no_grad()
    def integrate(self, pano_depth_map, pano_rgb, mask, pose):
        """Zero-copy, batched integration."""
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.torch_device)
            pose = torch.from_numpy(pose).float().to(self.torch_device)
            
        use_color = pano_rgb is not None
        if use_color and isinstance(pano_rgb, np.ndarray):
            pano_rgb = torch.from_numpy(pano_rgb).float().to(self.torch_device)

        # 1. BATCHED PYTORCH SAMPLING (All 6 faces in one GPU kernel)
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0).expand(6, -1, -1, -1)
        radial_depths = F.grid_sample(pano_depth_tensor, self.batched_grids, mode='bilinear', align_corners=True).squeeze(1)
        
        optical_depths = radial_depths * self.batched_z_mults
        optical_depths[optical_depths > self.max_depth] = 0.0
        
        color_faces = None
        if use_color:
            pano_rgb_tensor = pano_rgb.permute(2, 0, 1).unsqueeze(0).expand(6, -1, -1, -1)
            color_faces = F.grid_sample(pano_rgb_tensor, self.batched_grids, mode='bilinear', align_corners=True)
            color_faces = color_faces.permute(0, 2, 3, 1) / 255.0

        # 2. Sequential Open3D Insertion (Extrinsics mapped to CPU)
        for i in range(6):
            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            
            # Open3D strictly requires extrinsics on CPU
            extrinsic_np = torch.linalg.inv(face_pose).cpu().numpy()
            extrinsic_o3d = o3d.core.Tensor(extrinsic_np, o3d.core.float64, self.cpu_device)
            
            depth_dl = torch.utils.dlpack.to_dlpack(optical_depths[i].contiguous())
            depth_o3d = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(depth_dl))
            
            color_o3d = None
            if use_color:
                color_dl = torch.utils.dlpack.to_dlpack(color_faces[i].contiguous())
                color_o3d = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(color_dl))
                
            block_coords = self.vbg.compute_unique_block_coordinates(
                depth_o3d, self.intrinsic_o3d, extrinsic_o3d, depth_scale=1.0, depth_max=self.max_depth
            )
                        
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

    def extract_point_cloud(self):
        return self.vbg.extract_point_cloud(weight_threshold=3.0).to_legacy()

    def extract_mesh(self):
        print("[TSDF] Extracting high-density mesh...")
        t_mesh = self.vbg.extract_triangle_mesh(weight_threshold=3.0)
        return t_mesh.to_legacy()