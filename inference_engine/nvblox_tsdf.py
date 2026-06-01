import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import time

# --- FIX: Explicitly import the classes from their submodules ---
from nvblox_torch.mapper import Mapper
from nvblox_torch.camera import Camera
# ----------------------------------------------------------------

class NvbloxPanoTSDF:
    def __init__(self, voxel_size_m=0.01, max_depth=6.0, face_size=512, device="cuda"):
        self.device = device
        self.max_depth = max_depth
        self.face_size = face_size
        
        print(f"[TSDF] Initializing dynamic GPU nvblox Mapper (Voxel Size: {voxel_size_m}m)...")
        # nvblox dynamically allocates memory; no max_blocks required!
        # FIX: Use the imported Mapper class directly
        self.mapper = Mapper(voxel_size_m=voxel_size_m)
        
        # 1. Setup Cubemap Pinhole Intrinsics (90 Degree FOV)
        # focal_length = width / (2 * tan(FOV/2)) -> for 90 deg, f = width / 2
        f = self.face_size / 2.0
        c = self.face_size / 2.0
        # FIX: Use the imported Camera class directly
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
            torch.eye(3, device=self.device), # Front
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.device, dtype=torch.float32), # Right
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32), # Back
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32), # Left
            torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], device=self.device, dtype=torch.float32), # Top
            torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.device, dtype=torch.float32), # Bottom
        ]
        
        self.grids = []
        for d in self.face_dirs:
            d_norm = F.normalize(d, p=2, dim=-1)
            X, Y, Z = d_norm[..., 0], d_norm[..., 1], d_norm[..., 2]
            
            # Convert 3D ray to PanoVGGT Spherical UVs
            theta = torch.atan2(X, Z) 
            phi = torch.asin(torch.clamp(Y, -1.0, 1.0)) 
            
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            self.grids.append(torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0))

    @torch.no_grad()
    def integrate(self, pano_depth_map, pano_rgb, mask, pose):
        """Slices the pano output into 6 faces and integrates them into nvblox."""
        # Ensure tensor formats
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
            pose = torch.from_numpy(pose).float().to(self.device)
            
        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        
        # For each of the 6 faces of the cube...
        for i in range(6):
            # 1. Extract the pinhole depth image from the 360 map
            # Note: PanoVGGT outputs Euclidean radial distance. Pinhole cameras expect optical-Z depth.
            radial_depth = F.grid_sample(pano_depth_tensor, self.grids[i], mode='bilinear', align_corners=True).squeeze()
            
            # Convert radial distance to optical Z-axis depth (Z = r * cos(angle))
            z_multiplier = F.normalize(self.face_dirs[i], p=2, dim=-1)[..., 2].abs()
            optical_depth = radial_depth * z_multiplier
            
            # Filter depth by threshold
            optical_depth[optical_depth > self.max_depth] = 0.0
            
            # 2. Calculate the global pose for this specific face
            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            
            # 3. Integrate directly via zero-copy PyTorch bridge
            self.mapper.integrate_depth(
                optical_depth, 
                face_pose, 
                self.camera
            )

    def extract_mesh(self):
        """nvblox handles meshing natively, bypassing marching cubes bottlenecks."""
        print("[TSDF] Generating dense planar mesh from nvblox...")
        # nvblox natively generates and updates meshes from the TSDF volume
        return self.mapper.generate_mesh()