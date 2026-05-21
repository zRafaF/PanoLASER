import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF

# Ensure the PanoVGGT submodule is in the path
sys.path.append(os.path.abspath("./PanoVGGT"))

# Import the actual model class from the PanoVGGT submodule
from panovggt.models.panovggt_model import PanoVGGTModel

class PanoVGGTExtractor:
    def __init__(
        self, 
        config_path="PanoVGGT/training/config/default.yaml", 
        weights_path="checkpoints/model.pt", 
        device="cuda"
    ):
        self.device = device
        print(f"Loading PanoVGGT onto {self.device}...")
        
        # 1. Load and resolve configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find PanoVGGT config at {config_path}")
            
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        
        # 2. Initialize Model Architecture
        self.model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,
            enable_depth=mc.enable_depth,
            enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        ).to(self.device)
        
        # 3. Load Checkpoint
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
                
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[PanoVGGT] Missing keys (expected in partial load): {len(missing)}")
            
        self.model.eval()
        print("✅ PanoVGGT model loaded successfully.")

    @torch.no_grad()
    def process_frame(self, rgb_image: np.ndarray):
        """
        Runs a single RGB image through PanoVGGT to extract 3D geometry.
        """
        # 1. Preprocess: [0, 255] -> [0, 1]
        img_tensor = torch.from_numpy(rgb_image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # 2. Apply ImageNet Normalization (CRITICAL for DINOv2 backbone)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 3. Add batch and sequence dimensions: (B=1, N=1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 4. Inference with bfloat16 for efficiency
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.model(img_tensor)
            
        # 5. Extract Native Outputs & Transform Coordinate Space
        if "world_points" in preds and preds["world_points"] is not None:
            # The model already projected it into world space
            points_out = preds["world_points"].squeeze().cpu().float().numpy()
            depth_out = preds["depth"].squeeze().cpu().float().numpy() if "depth" in preds else torch.norm(preds["world_points"], dim=-1).squeeze().cpu().float().numpy()
        elif "local_points" in preds and preds["local_points"] is not None:
            # Manual transform: Local -> World using predicted pose
            local_pts = preds["local_points"].squeeze() # (H, W, 3)
            depth_out = torch.norm(local_pts, dim=-1).cpu().float().numpy()
            
            if "camera_poses" in preds and preds["camera_poses"] is not None:
                pose = preds["camera_poses"].squeeze()      # (4, 4)
                
                # Homogenize: (H, W, 4)
                ones = torch.ones((*local_pts.shape[:2], 1), device=local_pts.device)
                pts_homo = torch.cat([local_pts, ones], dim=-1)
                
                # Transform: pts_world = T * pts_local
                points_out = torch.matmul(pts_homo, pose.T)[..., :3].cpu().float().numpy()
            else:
                points_out = local_pts.cpu().float().numpy()
        else:
            raise RuntimeError("Model output did not contain 'world_points' or 'local_points'")
            
        conf_out = np.ones_like(depth_out)
        if "conf" in preds and preds["conf"] is not None:
             conf_out = preds["conf"].squeeze().cpu().float().numpy()

        return {
            "depth": depth_out,
            "conf": conf_out,
            "points": points_out
        }