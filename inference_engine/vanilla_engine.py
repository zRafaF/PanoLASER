import torch
import torch.nn as nn
import numpy as np
from .utils.geometry import unproject_equirectangular_to_points

class PanoVanillaEngine(nn.Module):
    def __init__(self, delegate: nn.Module):
        super().__init__()
        self.delegate = delegate
        self.device = next(delegate.parameters()).device

    @torch.no_grad()
    def forward(self, rgb_images_list):
        tensors = []
        for rgb_image in rgb_images_list:
            # Preprocess: [0, 255] -> [0, 1]
            t = torch.from_numpy(rgb_image).float() / 255.0
            t = t.permute(2, 0, 1)
            tensors.append(t)
            
        seq_tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.delegate(seq_tensor)
            
        depths = preds["depth"].squeeze(0).cpu().float().numpy()
        poses = preds["camera_poses"].squeeze(0).cpu().float().numpy()

        pts_list = [unproject_equirectangular_to_points(np.squeeze(d)) for d in depths]

        return {
            "depths": depths,
            "poses": poses,
            "points": pts_list
        }