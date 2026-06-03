import torch
import torch.nn.functional as F
import torch.utils.dlpack
import open3d as o3d
import numpy as np

class Open3DPanoVBG:
    def __init__(self, voxel_size_m=0.01, max_depth=6.0, face_size=512, device="cuda:0"):
        # Ensure device string is compatible with Open3D Core
        self.torch_device = torch.device(device)
        self.o3d_device = o3d.core.Device(device.upper())
        
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth
        self.face_size = face_size
        
        print(f"[TSDF] Initializing GPU Open3D VoxelBlockGrid (Voxel Size: {voxel_size_m}m)...")
        
        # Initialize the sparse Voxel Block Grid [cite: 1658]
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=['tsdf', 'weight', 'color'],
            attr_dtypes=[o3d.core.float32, o3d.core.float32, o3d.core.float32],
            attr_channels=[[1], [1], [3]],
            voxel_size=self.voxel_size_m,
            block_resolution=16,
            block_count=100000,
            device=self.o3d_device
        )
        
        # Intrinsics for a 90-degree FOV pinhole camera (Cubemap face)
        f = self.face_size / 2.0
        c = self.face_size / 2.0
        self.intrinsic_np = np.array([
            [f, 0, c],
            [0, f, c],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.intrinsic_o3d = o3d.core.Tensor(self.intrinsic_np, o3d.core.float64, self.o3d_device)
        self._precompute_cubemap_grids()

    def _precompute_cubemap_grids(self):
        """Precomputes the PyTorch sampling grids to slice the 360 image into 6 faces."""
        u = torch.linspace(-1, 1, self.face_size, device=self.torch_device)
        v = torch.linspace(-1, 1, self.face_size, device=self.torch_device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        self.face_dirs = [
            torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1),   # Front (+Z)
            torch.stack([torch.ones_like(u_grid), v_grid, -u_grid], dim=-1),  # Right (+X)
            torch.stack([-u_grid, v_grid, -torch.ones_like(u_grid)], dim=-1), # Back (-Z)
            torch.stack([-torch.ones_like(u_grid), v_grid, u_grid], dim=-1),  # Left (-X)
            torch.stack([u_grid, -torch.ones_like(u_grid), v_grid], dim=-1),  # Top (-Y)
            torch.stack([u_grid, torch.ones_like(u_grid), -v_grid], dim=-1)   # Bottom (+Y)
        ]
        
        self.face_rotations = [
            torch.eye(3, device=self.torch_device), # Front
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.torch_device, dtype=torch.float32), # Right
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.torch_device, dtype=torch.float32), # Back
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.torch_device, dtype=torch.float32), # Left
            torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=self.torch_device, dtype=torch.float32), # Top
            torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.torch_device, dtype=torch.float32), # Bottom
        ]
        
        self.grids = []
        for d in self.face_dirs:
            d_norm = F.normalize(d, p=2, dim=-1)
            X, Y, Z = d_norm[..., 0], d_norm[..., 1], d_norm[..., 2]
            
            theta = torch.atan2(X, Z) 
            phi = torch.asin(torch.clamp(Y, -1.0, 1.0)) 
            
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            self.grids.append(torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0))

    @torch.no_grad()
    def integrate(self, pano_depth_map, pano_rgb, mask, pose):
        """Zero-copy integration mapping PyTorch tensors directly to Open3D TSDF."""
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.torch_device)
            pose = torch.from_numpy(pose).float().to(self.torch_device)
            
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        
        use_color = pano_rgb is not None
        if use_color:
            if isinstance(pano_rgb, np.ndarray):
                pano_rgb = torch.from_numpy(pano_rgb).float().to(self.torch_device)
            pano_rgb_tensor = pano_rgb.permute(2, 0, 1).unsqueeze(0)
        
        for i in range(6):
            # 1. Extract pinhole depth
            radial_depth = F.grid_sample(pano_depth_tensor, self.grids[i], mode='bilinear', align_corners=True).squeeze()
            z_multiplier = F.normalize(self.face_dirs[i], p=2, dim=-1)[..., 2].abs()
            optical_depth = radial_depth * z_multiplier
            optical_depth[optical_depth > self.max_depth] = 0.0
            
            # 2. Extract pinhole color (Convert to [0, 1] float32 for O3D)
            color_face = None
            if use_color:
                color_face = F.grid_sample(pano_rgb_tensor, self.grids[i], mode='bilinear', align_corners=True).squeeze()
                color_face = color_face.permute(1, 2, 0).contiguous() / 255.0
            
            # 3. Calculate Global Pose for the Face
            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            
            # 4. Zero-Copy DLPack Transfer
            depth_dl = torch.utils.dlpack.to_dlpack(optical_depth.contiguous())
            depth_o3d = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(depth_dl))
            
            color_o3d = None
            if use_color:
                color_dl = torch.utils.dlpack.to_dlpack(color_face)
                color_o3d = o3d.t.geometry.Image(o3d.core.Tensor.from_dlpack(color_dl))
                
            extrinsic_o3d = o3d.core.Tensor(torch.linalg.inv(face_pose).cpu().numpy(), o3d.core.float64, self.o3d_device)
            
            # 5. Compute active blocks and integrate [cite: 1669, 1693, 1694]
            block_coords = self.vbg.compute_unique_block_coordinates(
                depth_o3d, self.intrinsic_o3d, extrinsic_o3d, depth_scale=1.0, depth_max=self.max_depth
            )
            
            # Insert active blocks into the sparse hash map [cite: 1671]
            self.vbg.hashmap().insert(block_coords, o3d.core.Tensor([1], o3d.core.uint8, self.o3d_device))
            
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
        """Extracts the dense point cloud directly from the TSDF volume."""
        return self.vbg.extract_point_cloud(weight_threshold=3.0).to_legacy()

    def extract_mesh(self):
        """Extracts the highly detailed Marching Cubes mesh[cite: 1685]."""
        print("[TSDF] Extracting high-density mesh...")
        t_mesh = self.vbg.extract_triangle_mesh(weight_threshold=3.0)
        return t_mesh.to_legacy()