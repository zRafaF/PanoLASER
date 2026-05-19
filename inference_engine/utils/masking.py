import numpy as np
import torch

def get_spherical_valid_mask(H: int, W: int, zenith_deg: float = 75.0, nadir_deg: float = -65.0) -> np.ndarray:
    """
    Creates a boolean mask for an equirectangular image, invalidating poles.
    Latitude ranges from +90 (top/zenith) to -90 (bottom/nadir).
    
    Returns:
        np.ndarray: A boolean mask of shape (H, W) where True means VALID.
    """
    # In an equirectangular image, row 0 is +90 deg, row H-1 is -90 deg
    latitudes = np.linspace(90.0, -90.0, H)
    
    # Create a 1D boolean array for valid rows
    valid_rows = (latitudes <= zenith_deg) & (latitudes >= nadir_deg)
    
    # Broadcast to a full 2D (H, W) mask
    mask = np.broadcast_to(valid_rows[:, None], (H, W))
    return mask

def apply_mask_to_tensor(tensor: torch.Tensor, mask: np.ndarray, fill_value=0.0) -> torch.Tensor:
    """
    Applies the numpy boolean mask to a PyTorch tensor (e.g., depth map or confidence map).
    """
    # Move mask to the same device as the tensor
    mask_tensor = torch.from_numpy(mask).to(tensor.device)
    
    # Expand mask dimensions if tensor has channel dims (e.g., N, C, H, W)
    while mask_tensor.dim() < tensor.dim():
        mask_tensor = mask_tensor.unsqueeze(0)
        
    return torch.where(
        mask_tensor, 
        tensor, 
        torch.tensor(fill_value, dtype=tensor.dtype, device=tensor.device)
    )