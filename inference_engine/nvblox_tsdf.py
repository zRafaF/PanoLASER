import torch
import torch.nn.functional as F
import numpy as np

# Correct imports matching the nvblox documentation
import nvblox
from nvblox.core.mapper import Mapper
from nvblox.core.camera import Camera
from nvblox.core.image import DepthImage
from nvblox.core.types import Transform, MemoryType

class NvbloxPanoTSDF:
    def __init__(self, voxel_size_m=0.01, max_depth=6.0, face_size=512, device="cuda"):
        self.device = device
        self.max_depth = max_depth
        self.face_size = face_size
        
        print(f"[TSDF] Initializing nvblox Mapper (Voxel Size: {voxel_size_m}m)...")
        
        # Initialize the Mapper and explicitly set memory to the GPU device
        self.mapper = Mapper(
            voxel_size_m=voxel_size_m, 
            memory_type=MemoryType.kDevice
        )
        
        # 1. Setup Cubemap Pinhole Intrinsics (90 Degree FOV)
        f = self.face_size / 2.0
        c = self.face_size / 2.0
        self.camera = Camera(f, f, c, c, self.face_size, self.face_size)
        
        # 2. Precompute Grid Tensors for Fast Unrolling
        self._precompute_cubemap_grids()

    def _precompute_cubemap_grids(self):
        """Precomputes the PyTorch sampling grids to instantly slice the 360 image into 6 faces."""
        u = torch.linspace(-1, 1, self.face_size, device=self.device)
        v = torch.linspace(-1, 1, self.face_size, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        # Define the 6 viewing directions (Front, Right, Back, Left, Top, Bottom)
        self.face_dirs = [
            torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1),   # Front (+Z)
            torch.stack([torch.ones_like(u_grid), v_grid, -u_grid], dim=-1),  # Right (+X)
            torch.stack([-u_grid, v_grid, -torch.ones_like(u_grid)], dim=-1), # Back (-Z)
            torch.stack([-torch.ones_like(u_grid), v_grid, u_grid], dim=-1),  # Left (-X)
            torch.stack([u_grid, -torch.ones_like(u_grid), v_grid], dim=-1),  # Top (-Y)
            torch.stack([u_grid, torch.ones_like(u_grid), -v_grid], dim=-1)   # Bottom (+Y)
        ]
        
        # Face rotation matrices relative to the main robot pose
        self.face_rotations = [
            torch.eye(3, device=self.device), 
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.device, dtype=torch.float32), 
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32), 
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32), 
            torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=self.device, dtype=torch.float32), 
            torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.device, dtype=torch.float32), 
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
        """Slices the pano output into 6 faces and integrates via nvblox.core."""
        # Ensure we are starting with tensors
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(self.device)
            
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        
        for i in range(6):
            # 1. Extract the pinhole depth image from the 360 map
            radial_depth = F.grid_sample(pano_depth_tensor, self.grids[i], mode='bilinear', align_corners=True).squeeze()
            
            # 2. Convert radial distance to optical Z-axis depth
            z_multiplier = F.normalize(self.face_dirs[i], p=2, dim=-1)[..., 2].abs()
            optical_depth = radial_depth * z_multiplier
            optical_depth[optical_depth > self.max_depth] = 0.0
            
            # 3. Format depth for nvblox.core bindings
            # Must convert to numpy float32, then wrap in DepthImage object
            depth_np = optical_depth.cpu().numpy().astype(np.float32)
            nvblox_depth = DepthImage(depth_np, memory_type=MemoryType.kDevice)
            
            # 4. Calculate global pose for this face and format for nvblox.core
            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            
            face_pose_np = face_pose.cpu().numpy().astype(np.float32)
            nvblox_pose = Transform(face_pose_np)
            
            # 5. Integrate!
            self.mapper.integrate_depth(
                nvblox_depth, 
                nvblox_pose, 
                self.camera
            )

    def extract_mesh(self):
        """nvblox handles meshing natively, outputting clean geometry."""
        print("[TSDF] Generating dense planar mesh from nvblox...")
        return self.mapper.generate_mesh()