import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
import torchvision.transforms.functional as TF

# Ensure the PanoVGGT submodule is in the path
sys.path.append(os.path.abspath("./PanoVGGT"))

# Import the actual model class from the submodule
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
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        print("✅ PanoVGGT model loaded successfully.")

    @torch.no_grad()
    def process_frame(self, rgb_image: np.ndarray):
        """
        Runs a single RGB image through PanoVGGT to extract geometry.
        """
        img_tensor = torch.from_numpy(rgb_image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # FIX 1: Apply Strict ImageNet Normalization so DINOv2 doesn't output flat depth
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.model(img_tensor)
            
        # Extract true radial depth map
        if "depth" in preds and preds["depth"] is not None:
            depth_out = preds["depth"].squeeze().cpu().float().numpy()
        elif "local_points" in preds and preds["local_points"] is not None:
            depth_out = torch.norm(preds["local_points"], dim=-1).squeeze().cpu().float().numpy()
        else:
            raise RuntimeError("Model output did not contain depth")
            
        conf_out = np.ones_like(depth_out)
        if "conf" in preds and preds["conf"] is not None:
             conf_out = preds["conf"].squeeze().cpu().float().numpy()

        return {
            "depth": depth_out,
            "conf": conf_out
        }