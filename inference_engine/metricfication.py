import numpy as np
import open3d as o3d

def estimate_metric_scale_from_floor(local_pts, target_camera_height=1.7, normal_tolerance_deg=15.0, inlier_threshold=0.3):
    """
    Finds the floor plane in the local camera point cloud and calculates the absolute metric scale.
    Assumes an OpenCV camera coordinate system (Y is DOWN).
    """
    # 1. Flatten the point cloud to (N, 3) in case it comes directly from the (H, W, 3) image output
    local_pts = local_pts.reshape(-1, 3)
    
    # 2. Drop the "Hole": Filter out the [0, 0, 0] points caused by the nadir mask 
    # so they don't skew the RANSAC calculation
    valid_mask = np.linalg.norm(local_pts, axis=-1) > 0.1
    local_pts = local_pts[valid_mask]
    
    # 3. Isolate the lower hemisphere (Y > 0.1 in OpenCV space)
    lower_pts = local_pts[local_pts[:, 1] > 0.1]
    
    if len(lower_pts) < 100:
        return None, 0.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lower_pts)
    
    # 4. RANSAC Plane Fitting
    # This will easily span across any masked "holes" directly beneath the camera.
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                                 ransac_n=3,
                                                 num_iterations=200)
    except Exception:
        return None, 0.0
        
    [a, b, c, d] = plane_model
    
    # 5. Check if the plane is horizontal (Normal aligned with Y-axis)
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    y_axis = np.array([0, 1, 0])
    dot_product = np.clip(np.abs(np.dot(normal, y_axis)), 0.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    
    # 6. Calculate Confidence (Percentage of lower points belonging to the floor)
    confidence = len(inliers) / float(len(lower_pts))
    
    if angle <= normal_tolerance_deg and confidence >= inlier_threshold:
        # Distance from camera origin (0,0,0) to the plane
        estimated_height = abs(d)
        if estimated_height < 0.1: # Prevent division by zero anomalies
            return None, confidence
            
        scale_factor = target_camera_height / estimated_height
        return scale_factor, confidence
    else:
        # Not a flat horizontal floor (e.g., stairs, clutter, or walking outside on uneven terrain)
        return None, confidence