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
        
        self.means = nn.Parameter(torch.empty((0, 3), device=self.device))
        self.scales = nn.Parameter(torch.empty((0, 3), device=self.device))
        self.quats = nn.Parameter(torch.empty((0, 4), device=self.device))
        self.opacities = nn.Parameter(torch.empty((0,), device=self.device))
        self.colors = nn.Parameter(torch.empty((0, 3), device=self.device)) 
        
        self.optimizer = None
        self._precompute_cubemap_cameras()
        
    def _precompute_cubemap_cameras(self):
        focal = self.face_size / 2.0
        c = self.face_size / 2.0
        self.K = torch.tensor([
            [focal, 0, c],
            [0, focal, c],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
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
            norm = torch.norm(ray_global, dim=-1)
            theta = torch.atan2(ray_global[..., 0], ray_global[..., 2]) 
            phi = torch.asin(torch.clamp(ray_global[..., 1] / norm, -1.0, 1.0)) 
            grids.append(torch.stack([theta / torch.pi, phi / (torch.pi / 2.0)], dim=-1))
            
        self.batched_grids = torch.stack(grids, dim=0)

    @torch.no_grad()
    def seed_new_points(self, points_np, colors_np):
        num_new = points_np.shape[0]
        if num_new == 0: return
        
        new_means = torch.from_numpy(points_np).float().to(self.device)
        new_colors = torch.from_numpy(colors_np).float().to(self.device)
        
        dist_to_origin = torch.norm(new_means, dim=1, keepdim=True)
        # Small initialization so optimizer has to work to expand them
        new_scales = torch.log(torch.clamp(dist_to_origin * 0.003, min=0.0005)).repeat(1, 3) 
        
        new_quats = torch.zeros((num_new, 4), device=self.device)
        new_quats[:, 0] = 1.0 
        
        new_opacities = torch.zeros((num_new,), device=self.device) 
        
        self.means = nn.Parameter(torch.cat([self.means, new_means], dim=0))
        self.scales = nn.Parameter(torch.cat([self.scales, new_scales], dim=0))
        self.quats = nn.Parameter(torch.cat([self.quats, new_quats], dim=0))
        self.opacities = nn.Parameter(torch.cat([self.opacities, new_opacities], dim=0))
        self.colors = nn.Parameter(torch.cat([self.colors, new_colors], dim=0))
        
        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': 0.0001},
            {'params': [self.colors], 'lr': 0.01},
            {'params': [self.scales, self.quats], 'lr': 0.002}, 
            {'params': [self.opacities], 'lr': 0.02}            
        ])

    def train_submap(self, batch_rgbs, batch_poses, iterations=15):
        if self.means.shape[0] == 0: return
        
        self.train()
        gt_cubemaps = []
        viewmats = []
        
        with torch.no_grad():
            for rgb, pose in zip(batch_rgbs, batch_poses):
                img_t = torch.from_numpy(rgb).float().to(self.device) / 255.0
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
                
                faces = []
                for i in range(4): 
                    face = F.grid_sample(img_t, self.batched_grids[i:i+1], mode='bilinear', align_corners=False)
                    faces.append(face.squeeze(0).permute(1, 2, 0)) 
                    
                    pose_t = torch.from_numpy(pose).float().to(self.device)
                    R_c_w = pose_t[:3, :3].T
                    t_c_w = -R_c_w @ pose_t[:3, 3]
                    
                    face_viewmat = torch.eye(4, device=self.device)
                    face_viewmat[:3, :3] = self.face_rotations[i].T @ R_c_w
                    face_viewmat[:3, 3] = self.face_rotations[i].T @ t_c_w
                    viewmats.append(face_viewmat)
                    
                gt_cubemaps.extend(faces)
                
        viewmats = torch.stack(viewmats) 
        K_batched = self.K.unsqueeze(0).expand(len(viewmats), -1, -1)
        
        for step in range(iterations):
            self.optimizer.zero_grad()
            renders, _, _ = rasterization(
                means=self.means, quats=self.quats, scales=torch.exp(self.scales),
                opacities=torch.sigmoid(self.opacities), colors=torch.sigmoid(self.colors),
                viewmats=viewmats, Ks=K_batched, width=self.face_size, height=self.face_size, packed=False
            )
            
            loss = 0.0
            for r_img, gt_img in zip(renders, gt_cubemaps): loss += F.l1_loss(r_img, gt_img)
            loss = loss / len(renders)
            loss.backward()
            self.optimizer.step()
            
    @torch.no_grad()
    def get_o3d_pointcloud(self):
        pcd = o3d.geometry.PointCloud()
        if self.means.shape[0] == 0: return pcd
        valid_mask = torch.sigmoid(self.opacities) > 0.05
        if valid_mask.sum() == 0: return pcd
        
        pcd.points = o3d.utility.Vector3dVector(self.means[valid_mask].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(torch.sigmoid(self.colors[valid_mask]).cpu().numpy())
        return pcd

    @torch.no_grad()
    def save_ply(self, path):
        import plyfile
        if self.means.shape[0] == 0: return

        valid_mask = torch.sigmoid(self.opacities) > 0.05
        xyz = self.means[valid_mask].cpu().numpy()
        normals = np.zeros_like(xyz)
        
        rgb = torch.sigmoid(self.colors[valid_mask]).cpu().numpy()
        f_dc = (rgb - 0.5) / 0.28209479177387814
        
        opacities = self.opacities[valid_mask].unsqueeze(-1).cpu().numpy()
        scales = self.scales[valid_mask].cpu().numpy()
        quats = F.normalize(self.quats[valid_mask], p=2, dim=-1).cpu().numpy()
        
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
        elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
        elements['opacity'] = opacities[:, 0]
        elements['scale_0'], elements['scale_1'], elements['scale_2'] = scales[:, 0], scales[:, 1], scales[:, 2]
        elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        el = plyfile.PlyElement.describe(elements, 'vertex')
        plyfile.PlyData([el]).write(path)