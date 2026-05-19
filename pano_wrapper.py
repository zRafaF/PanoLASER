import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

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
        
        # 1. Load Configuration
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
        
        # 3. Load Checkpoint (handling state_dict nesting and module prefixes)
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
                
        # Clean state dict keys (remove 'module.' if it was saved via DataParallel)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[PanoVGGT] Missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
            
        self.model.eval()
        print("✅ PanoVGGT model loaded successfully.")

    @torch.no_grad()
    def process_frame(self, rgb_image: np.ndarray):
        """
        Runs a single RGB image through PanoVGGT to extract geometry.
        Args:
            rgb_image: (H, W, 3) uint8 numpy array in RGB format.
        """
        # 1. Preprocess image (H, W, C) -> (C, H, W) -> float [0, 1]
        img_tensor = torch.from_numpy(rgb_image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        
        # PanoVGGT expects Batch and Sequence dimensions: (B, N, C, H, W)
        # B=1 (Batch), N=1 (Single frame)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 2. Run Inference using bfloat16 mixed precision (as done in app.py)
        # Note: If your GPU does not support bfloat16, fallback to float16
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.model(img_tensor)
            
        # 3. Extract Radial Depth
        # Following app.py logic: prefer 'local_points' magnitude, fallback to 'depth'
        if "local_points" in preds and preds["local_points"] is not None:
            lp = preds["local_points"]
            radial = torch.norm(lp, dim=-1)
        elif "depth" in preds and preds["depth"] is not None:
            radial = preds["depth"].clone()
        else:
            raise RuntimeError("Model output did not contain 'local_points' or 'depth'")
            
        # 4. Format Outputs (remove Batch/Seq dims, move to CPU numpy)
        depth_out = radial.squeeze().cpu().float().numpy()
        
        # Attempt to grab confidence if the model outputs it, otherwise default to 1.0
        conf_out = np.ones_like(depth_out)
        if "conf" in preds and preds["conf"] is not None:
             conf_out = preds["conf"].squeeze().cpu().float().numpy()

        return {
            "depth": depth_out,
            "conf": conf_out
        }