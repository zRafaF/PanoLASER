import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
from .utils.geometry import unproject_equirectangular_to_points

class PanoVanillaEngine(nn.Module):
    def __init__(self, delegate: nn.Module):
        """
        Args:
            delegate: The raw, initialized PanoVGGTModel 
        """
        super().__init__()
        self.delegate = delegate
        # Find the device the model is loaded on
        self.device = next(delegate.parameters()).device

    @torch.no_grad()
    def forward(self, rgb_images_list):
        """
        Processes a temporal sequence of RGB images.
        Returns pure depth maps, relative poses, and unprojected raw 3D points.
        """
        tensors = []
        for rgb_image in rgb_images_list:
            # Preprocess: [0, 255] -> [0, 1]
            t = torch.from_numpy(rgb_image).float() / 255.0
            t = t.permute(2, 0, 1)
            # DINOv2 ImageNet Normalization (Required for Geometry extraction)
            t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors.append(t)
            
        # Stack to (N, C, H, W) and add Batch dimension -> (1, N, C, H, W)
        seq_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.delegate(seq_tensor)
            
        # Extract sequences and remove batch dimension
        depths = preds["depth"].squeeze(0).cpu().float().numpy()      # (N, H, W)
        poses = preds["camera_poses"].squeeze(0).cpu().float().numpy() # (N, 4, 4)

        # Unproject all depths into native Cartesian points automatically
        pts_list = [unproject_equirectangular_to_points(d) for d in depths]

        return {
            "depths": depths,
            "poses": poses,
            "points": pts_list
        }