import numpy as np

def unproject_equirectangular_to_points(depth_map: np.ndarray) -> np.ndarray:
    """
    Converts an equirectangular radial depth map into 3D Cartesian coordinates.
    Assumes OpenCV coordinate system (X=Right, Y=Down, Z=Forward).
    
    Args:
        depth_map: (H, W) array of radial distances from the camera center.
    Returns:
        points: (H, W, 3) array of 3D coordinates.
    """
    H, W = depth_map.shape
    
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert pixels to spherical angles
    # Longitude (theta): -pi to +pi (Left to Right)
    theta = (u / W - 0.5) * 2 * np.pi
    
    # Latitude (phi): -pi/2 to +pi/2 (Top to Bottom, Y is down in CV)
    phi = (v / H - 0.5) * np.pi
    
    # Spherical to Cartesian Transformation
    X = depth_map * np.cos(phi) * np.sin(theta)
    Y = depth_map * np.sin(phi)
    Z = depth_map * np.cos(phi) * np.cos(theta)
    
    # Stack into a 3D point array (H, W, 3)
    return np.stack([X, Y, Z], axis=-1)