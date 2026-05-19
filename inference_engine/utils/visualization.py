import cv2
import numpy as np

def visualize_polar_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Dims and tints the excluded regions of the image to visualize the mask.
    
    Args:
        image: (H, W, 3) uint8 RGB image.
        mask: (H, W) boolean mask where True is the VALID region.
    """
    vis_image = image.copy()
    
    # Create a darkened version of the image for the excluded parts
    darkened = (vis_image * 0.4).astype(np.uint8)
    
    # Apply a slight red tint to the darkened area to make it obvious
    red_tint = np.zeros_like(vis_image)
    red_tint[:, :, 0] = 80  # Add to Red channel (Assuming RGB format from Gradio)
    darkened = cv2.add(darkened, red_tint)
    
    # The mask is True where VALID. We want to apply the tint where INVALID (~mask)
    invalid_mask = ~mask
    vis_image[invalid_mask] = darkened[invalid_mask]
    
    return vis_image

def visualize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Converts a 1-channel depth map into a colorized heatmap for visualization.
    """
    # Normalize depth to 0-255
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    
    # Apply a colormap (PLASMA is great for depth)
    colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    # OpenCV outputs BGR, Gradio expects RGB
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)