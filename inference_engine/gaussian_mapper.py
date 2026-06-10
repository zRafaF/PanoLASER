import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from gsplat.rendering import rasterization

class PanoGaussianMapper(nn.Module):
    def __init__(self, device="cuda", face_size=512):
        super().__init__()
        self.device = torch.device(device)
        self.face_size = face_size
        
        # Gaussian Parameters
        self.means = nn.Parameter(torch.empty((0, 3), device=self.device))
        self.scales = nn.Parameter(torch.empty((0, 3), device=self.device))
        self.quats = nn.Parameter(torch.empty((0, 4), device=self.device))
        self.opacities = nn.Parameter(torch.empty((0,), device=self.device))
        self.colors = nn.Parameter(torch.empty((0, 3), device=self.device)) # Base RGB (SH0)
        
        self.optimizer = None
        self._precompute_cubemap_cameras()
        
    def _precompute_cubemap_cameras(self):
        """Generates the 6 Pinhole Cameras for a 360 Panorama"""
        # 90-degree FOV pinhole math
        focal = self.face_size / 2.0
        c = self.face_size / 2.0
        self.K = torch.tensor([
            [focal, 0, c],
            [0, focal, c],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Grid sample coordinates for Equirectangular -> Cubemap extraction
        u = torch.linspace(-1, 1, self.face_size, device=self.device)
        v = torch.linspace(-1, 1, self.face_size, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        base_rays = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
        
        self.face_rotations = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device, dtype=torch.float32),  # Front
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.device, dtype=torch.float32), # Right
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32),# Back
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32), # Left
            torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], device=self.device, dtype=torch.float32),# Top
            torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], device=self.device, dtype=torch.float32), # Bottom
        ]
        
        grids = []
        for R in self.face_rotations:
            ray_global = base_rays @ R.T
            norm = torch.norm(ray_global, dim=-1)
            theta = torch.atan2(ray_global[..., 0], ray_global[..., 2]) 
            phi = torch.asin(torch.clamp(ray_global[..., 1] / norm, -1.0, 1.0)) 
            grids.append(torch.stack([theta / torch.pi, phi / (torch.pi / 2.0)], dim=-1))
            
        self.batched_grids = torch.stack(grids, dim=0)

    @torch.no_grad()
    def seed_new_points(self, points_np, colors_np):
        """Adds new VGGT points to the Gaussian map"""
        num_new = points_np.shape[0]
        if num_new == 0: return
        
        new_means = torch.from_numpy(points_np).float().to(self.device)
        new_colors = torch.from_numpy(colors_np).float().to(self.device)
        
        # Initialize small isotropic scales and default opacities
        dist_to_origin = torch.norm(new_means, dim=1, keepdim=True)
        # Scale heuristic: 1% of distance to camera, roughly representing point footprint
        new_scales = torch.log(torch.clamp(dist_to_origin * 0.01, min=0.001)).repeat(1, 3) 
        
        new_quats = torch.zeros((num_new, 4), device=self.device)
        new_quats[:, 0] = 1.0 # Identity quaternion (W, X, Y, Z)
        
        # Start opacities at 0.5 (logit scale: inv_sigmoid(0.5) = 0)
        new_opacities = torch.zeros((num_new,), device=self.device) 
        
        # Concatenate with existing state
        self.means = nn.Parameter(torch.cat([self.means, new_means], dim=0))
        self.scales = nn.Parameter(torch.cat([self.scales, new_scales], dim=0))
        self.quats = nn.Parameter(torch.cat([self.quats, new_quats], dim=0))
        self.opacities = nn.Parameter(torch.cat([self.opacities, new_opacities], dim=0))
        self.colors = nn.Parameter(torch.cat([self.colors, new_colors], dim=0))
        
        # Re-initialize Adam with new parameter shapes
        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': 0.0001},
            {'params': [self.colors], 'lr': 0.01},
            {'params': [self.scales, self.quats], 'lr': 0.005},
            {'params': [self.opacities], 'lr': 0.05}
        ])

    def train_submap(self, batch_rgbs, batch_poses, iterations=15):
        """Runs fast gradient descent to align Gaussians with the panoramic RGBs"""
        if self.means.shape[0] == 0: return
        
        self.train()
        
        # Prepare GT Cubemaps on GPU
        gt_cubemaps = []
        viewmats = []
        
        with torch.no_grad():
            for rgb, pose in zip(batch_rgbs, batch_poses):
                # 1. Image to GPU
                img_t = torch.from_numpy(rgb).float().to(self.device) / 255.0
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                
                # 2. Extract 6 faces (Ignoring Top/Bottom poles for training stability)
                faces = []
                for i in range(4): # Just train on Front, Right, Back, Left
                    face = F.grid_sample(img_t, self.batched_grids[i:i+1], mode='bilinear', align_corners=False)
                    faces.append(face.squeeze(0).permute(1, 2, 0)) # (H, W, 3)
                    
                    # 3. Calculate View Matrix (Extrinsics) for each face
                    # FIX: Explicitly cast the NumPy pose array to a PyTorch GPU Tensor
                    pose_t = torch.from_numpy(pose).float().to(self.device)
                    
                    R_c_w = pose_t[:3, :3].T
                    t_c_w = -R_c_w @ pose_t[:3, 3]
                    
                    face_viewmat = torch.eye(4, device=self.device)
                    face_viewmat[:3, :3] = self.face_rotations[i].T @ R_c_w
                    face_viewmat[:3, 3] = self.face_rotations[i].T @ t_c_w
                    viewmats.append(face_viewmat)
                    
                gt_cubemaps.extend(faces)
                
        viewmats = torch.stack(viewmats) # (N*4, 4, 4)
        K_batched = self.K.unsqueeze(0).expand(len(viewmats), -1, -1)
        
        # Training Loop
        for step in range(iterations):
            self.optimizer.zero_grad()
            
            # gsplat 1.0+ Rasterization
            renders, _, _ = rasterization(
                means=self.means,
                quats=self.quats,
                scales=torch.exp(self.scales),
                opacities=torch.sigmoid(self.opacities),
                colors=torch.sigmoid(self.colors),
                viewmats=viewmats,
                Ks=K_batched,
                width=self.face_size,
                height=self.face_size,
                packed=False
            )
            
            # Compute L1 Loss against Ground Truth
            loss = 0.0
            for r_img, gt_img in zip(renders, gt_cubemaps):
                loss += F.l1_loss(r_img, gt_img)
                
            loss = loss / len(renders)
            loss.backward()
            self.optimizer.step()
            
    @torch.no_grad()
    def get_o3d_pointcloud(self):
        """Extracts current Gaussian Means as a dense point cloud for Gradio UI"""
        pcd = o3d.geometry.PointCloud()
        if self.means.shape[0] == 0: return pcd
        
        # Filter out completely invisible Gaussians
        valid_mask = torch.sigmoid(self.opacities) > 0.1
        
        pcd.points = o3d.utility.Vector3dVector(self.means[valid_mask].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(torch.sigmoid(self.colors[valid_mask]).cpu().numpy())
        return pcd

    @torch.no_grad()
    def save_ply(self, path):
        """Exports standard 3DGS .ply for WebGL viewers (SuperSplat, PlayCanvas)"""
        pass