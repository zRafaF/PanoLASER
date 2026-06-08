import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F

class HighResTextureBaker:
    def __init__(self, face_size=1024, device="cuda"):
        """
        Takes a raw geometry mesh and bakes high-res panoramas onto a UV-unwrapped texture atlas.
        face_size: Resolution of the individual pinhole projections (higher = sharper textures).
        """
        self.face_size = face_size
        self.device = torch.device(device)
        self._precompute_cubemap_grids()
        
        # Pinhole camera intrinsics for a 90-degree FOV cubemap face
        self.focal_length = self.face_size / 2.0
        self.principal_point = self.face_size / 2.0
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.face_size, self.face_size, 
            self.focal_length, self.focal_length, 
            self.principal_point, self.principal_point
        )

    def _precompute_cubemap_grids(self):
        u = torch.linspace(-1, 1, self.face_size, device=self.device)
        v = torch.linspace(-1, 1, self.face_size, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        base_rays = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
        
        self.face_rotations = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32), 
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32),
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32),
            np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
        ]
        
        grids = []
        for R in self.face_rotations:
            R_tensor = torch.from_numpy(R).to(self.device)
            ray_global = base_rays @ R_tensor.T
            X, Y, Z = ray_global[..., 0], ray_global[..., 1], ray_global[..., 2]
            norm = torch.sqrt(X**2 + Y**2 + Z**2)
            theta = torch.atan2(X, Z) 
            phi = torch.asin(torch.clamp(Y / norm, -1.0, 1.0)) 
            u_norm = theta / torch.pi
            v_norm = phi / (torch.pi / 2.0)
            grids.append(torch.stack([u_norm, v_norm], dim=-1))
            
        self.batched_grids = torch.stack(grids, dim=0)

    def extract_pinhole_images(self, pano_rgb_np):
        pano_tensor = torch.from_numpy(pano_rgb_np).float().to(self.device)
        pano_tensor = pano_tensor.permute(2, 0, 1).unsqueeze(0)
        
        faces = []
        for i in range(6):
            if i == 5: # Skip the bottom face (tripod/nadir)
                continue 
                
            face_tensor = F.grid_sample(
                pano_tensor, self.batched_grids[i:i+1], mode='bilinear', align_corners=True
            ).squeeze(0)
            
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            face_np_contiguous = np.ascontiguousarray(face_np)
            
            faces.append((face_np_contiguous, self.face_rotations[i]))
            
        return faces

    def run_baking_pass(self, raw_mesh, sequence_frames, global_poses):
        print("\n[Texture Baker] Starting Offline High-Res UV Unwrapping & Baking...")
        
        # --- 1. MESH QUALITY ENHANCEMENT ---
        print("[Texture Baker] Enhancing mesh topology (Smoothing & Subdividing)...")
        # Subdivide to increase triangle count (makes the mesh denser for better texturing)
        raw_mesh = raw_mesh.subdivide_midpoint(number_of_iterations=1)
        # Taubin smoothing removes the blocky 'staircase' voxel artifacts without shrinking the room
        raw_mesh = raw_mesh.filter_smooth_taubin(number_of_iterations=20)
        
        raw_mesh.compute_vertex_normals()
        # Clear existing low-res vertex colors so Open3D builds an image texture atlas
        raw_mesh.vertex_colors = o3d.utility.Vector3dVector() 
        
        camera_trajectory = o3d.camera.PinholeCameraTrajectory()
        images = []
        
        # We don't need every single frame for texturing (saves massive RAM)
        stride = max(1, len(sequence_frames) // 30) 
        
        for idx in range(0, len(sequence_frames), stride):
            pano_rgb = sequence_frames[idx]
            base_pose = global_poses[idx]
            
            faces = self.extract_pinhole_images(pano_rgb)
            
            for face_img, face_R in faces:
                o3d_color = o3d.geometry.Image(face_img)
                
                # --- THE FIX: Dummy Depth Map ---
                # Open3D's optimizer explicitly requires an RGBDImage format.
                # We create a dummy flat depth map to satisfy the API.
                dummy_depth = np.ones((self.face_size, self.face_size), dtype=np.float32) * 2.0
                o3d_depth = o3d.geometry.Image(dummy_depth)
                
                o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth, 
                    depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                
                images.append(o3d_rgbd)
                
                face_pose = base_pose.copy()
                face_pose[:3, :3] = base_pose[:3, :3] @ face_R
                extrinsic = np.linalg.inv(face_pose)
                
                cam_param = o3d.camera.PinholeCameraParameters()
                cam_param.intrinsic = self.intrinsic
                cam_param.extrinsic = extrinsic
                camera_trajectory.parameters.append(cam_param)

        print(f"[Texture Baker] Baking {len(images)} high-res pinhole views onto the mesh...")
        
        # Loosen the depth visibility checks so our dummy depth map is accepted
        options = o3d.pipelines.color_map.RigidOptimizerOption()
        options.maximum_allowable_depth = 10.0
        options.depth_threshold_for_visiblity_check = 10.0
        
        o3d.pipelines.color_map.run_rigid_optimizer(
            raw_mesh, 
            images, 
            camera_trajectory, 
            options
        )
        
        print("[Texture Baker] Baking Complete!")
        return raw_mesh