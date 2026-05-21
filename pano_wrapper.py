import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF

# Ensure the PanoVGGT submodule is in the path
sys.path.append(os.path.abspath("./PanoVGGT"))
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
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find PanoVGGT config at {config_path}")
            
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        
        self.model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,
            enable_depth=mc.enable_depth,
            enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        ).to(self.device)
        
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
                
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        print("✅ PanoVGGT model loaded successfully.")

    @torch.no_grad()
    def process_window(self, rgb_images_list: list):
        """
        Runs a temporal sequence of RGB images through PanoVGGT to extract 
        joint geometry and relative camera poses.
        """
        tensors = []
        for rgb_image in rgb_images_list:
            t = torch.from_numpy(rgb_image).float() / 255.0
            t = t.permute(2, 0, 1)
            t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors.append(t)
            
        # Stack to (N, C, H, W) and add Batch dimension -> (1, N, C, H, W)
        seq_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.model(seq_tensor)
            
        # Extract sequences
        depth_out = preds["depth"].squeeze(0).cpu().float().numpy()      # (N, H, W)
        poses_out = preds["camera_poses"].squeeze(0).cpu().float().numpy() # (N, 4, 4)

        return {
            "depths": depth_out,
            "poses": poses_out
        }