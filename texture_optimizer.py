import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F

class HighResTextureBaker:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)

    @torch.no_grad()
    def run_baking_pass(self, raw_mesh, sequence_frames, global_poses):
        print("\n[Texture Baker] Starting Pure PyTorch High-Res Vertex Projection...")
        
        # --- 1. MESH TOPOLOGY ENHANCEMENT ---
        print("[Texture Baker] Subdividing mesh for ultra-dense geometry...")
        # 2 subdivisions turns 1 triangle into 16, exponentially increasing density
        dense_mesh = raw_mesh.subdivide_midpoint(number_of_iterations=2)
        # Taubin smoothing removes blocky artifacts without shrinking the total volume
        dense_mesh = dense_mesh.filter_smooth_taubin(number_of_iterations=20)
        dense_mesh.compute_vertex_normals()
        
        # Move geometry to GPU
        vertices = torch.from_numpy(np.asarray(dense_mesh.vertices)).float().to(self.device)
        normals = torch.from_numpy(np.asarray(dense_mesh.vertex_normals)).float().to(self.device)
        
        num_verts = vertices.shape[0]
        print(f"[Texture Baker] Mesh density increased to {num_verts} vertices!")
        
        # Buffers to average overlapping frames
        color_accum = torch.zeros((num_verts, 3), device=self.device, dtype=torch.float32)
        weight_accum = torch.zeros((num_verts, 1), device=self.device, dtype=torch.float32)
        
        # Sub-sample to save time (we don't need 30 overlapping photos of the same wall)
        stride = max(1, len(sequence_frames) // 30)
        frames_to_process = list(range(0, len(sequence_frames), stride))
        
        print(f"[Texture Baker] GPU Raycasting {len(frames_to_process)} panoramas...")
        
        for idx in frames_to_process:
            pano_np = sequence_frames[idx]
            pose_np = global_poses[idx]
            
            # [H, W, 3] -> [1, 3, H, W]
            pano_t = torch.from_numpy(pano_np).float().to(self.device) / 255.0
            pano_t = pano_t.permute(2, 0, 1).unsqueeze(0)
            
            pose_t = torch.from_numpy(pose_np).float().to(self.device)
            
            # Extract inverse transform (world -> camera)
            R_inv = pose_t[:3, :3].T
            t_inv = -R_inv @ pose_t[:3, 3]
            
            # Transform vertices to local camera frame
            v_local = (vertices @ R_inv.T) + t_inv
            
            # Distance filter (only project onto geometry within 4 meters)
            dist = torch.norm(v_local, dim=-1, keepdim=True)
            valid_dist = (dist > 0.1) & (dist < 4.0)
            
            # Normal filter (ensure the face is pointing towards the camera)
            ray_dir = F.normalize(vertices - pose_t[:3, 3], dim=-1)
            facing_cam = (ray_dir * normals).sum(dim=-1, keepdim=True) < -0.1
            
            mask = valid_dist & facing_cam
            if not mask.any():
                continue
            
            # Equirectangular UV mapping (OpenCV coordinate frame: X right, Y down, Z forward)
            X = v_local[:, 0:1]
            Y = v_local[:, 1:2]
            Z = v_local[:, 2:3]
            
            theta = torch.atan2(X, Z) # Longitude: [-pi, pi]
            phi = torch.asin(torch.clamp(Y / dist, -1.0, 1.0)) # Latitude: [-pi/2, pi/2]
            
            # Map into the [-1, 1] range required by grid_sample
            u = theta / np.pi
            v = phi / (np.pi / 2.0)
            
            grid = torch.cat([u, v], dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Sample all N vertices in one instantaneous shot
            sampled = F.grid_sample(pano_t, grid, mode='bilinear', align_corners=True)
            colors = sampled.squeeze().T # [N, 3]
            
            # Weight the sampled color: closer distance = higher priority
            weight = mask.float() * (1.0 / (dist + 1e-5))
            
            color_accum += colors * weight
            weight_accum += weight
            
        print("[Texture Baker] Normalizing color blending...")
        # Avoid zero division and calculate final blended color
        final_colors = color_accum / torch.clamp(weight_accum, min=1e-5)
        final_colors = torch.clamp(final_colors, 0.0, 1.0)
        
        # Inject back into the Open3D mesh
        dense_mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors.cpu().numpy())
        
        print("[Texture Baker] Pipeline Complete!")
        return dense_mesh