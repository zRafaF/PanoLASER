import gradio as gr
import numpy as np
import open3d as o3d
import tempfile
import os
from PIL import Image

from inference_engine.utils.masking import get_spherical_valid_mask
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from inference_engine.utils.geometry import unproject_equirectangular_to_points
from pano_wrapper import PanoVGGTExtractor

# Initialize the model wrapper globally
extractor = PanoVGGTExtractor()

def create_point_cloud_ply(rgb_image, depth_map, mask):
    """Converts depth/rgb into a 3D PLY file and returns the file path."""
    # 1. Get 3D Coordinates
    xyz_points = unproject_equirectangular_to_points(depth_map)
    
    # 2. Flatten arrays and apply the valid mask
    valid_mask_flat = mask.flatten()
    points_flat = xyz_points.reshape(-1, 3)[valid_mask_flat]
    colors_flat = rgb_image.reshape(-1, 3)[valid_mask_flat] / 255.0  # Open3D expects 0-1 colors
    
    # 3. Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)
    
    # Optional: Downsample for smoother browser performance
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    # 4. Save to a temporary .ply file for Gradio
    temp_dir = tempfile.mkdtemp()
    ply_path = os.path.join(temp_dir, "reconstruction.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    
    return ply_path

def process_pipeline(input_image_pil, zenith_limit, nadir_limit):
    if input_image_pil is None:
        return None, None, None
    
    input_image = np.array(input_image_pil)
    H, W = input_image.shape[:2]
    
    # Masking
    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    masked_rgb_vis = visualize_polar_mask(input_image, mask)
    
    # Inference
    predictions = extractor.process_frame(input_image)
    depth_map = predictions["depth"]
    
    # Apply Mask to depth (Zero out the poles)
    depth_map[~mask] = 0.0
    depth_vis = visualize_depth(depth_map)
    
    # Generate 3D Point Cloud file
    ply_file_path = create_point_cloud_ply(input_image, depth_map, mask)
    
    return Image.fromarray(masked_rgb_vis), Image.fromarray(depth_vis), ply_file_path

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="PanoLASER Streaming Engine") as demo:
    gr.Markdown("# 🌐 PanoLASER: Spherical Masking & Inference Sandbox")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Input 360° Image", type="pil")
            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-60, step=1, label="Nadir Limit")
            run_btn = gr.Button("Process Frame & Generate 3D", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("2D Masking Analysis"):
                    output_rgb = gr.Image(label="Masked Input", type="pil")
                    output_depth = gr.Image(label="Masked Depth Map Prediction", type="pil")
                
                with gr.Tab("3D Point Cloud View"):
                    output_3d = gr.Model3D(label="Dense Point Cloud Reconstruction", clear_color=[0.1, 0.1, 0.1, 1.0])

    run_btn.click(
        fn=process_pipeline,
        inputs=[input_img, zenith_slider, nadir_slider],
        outputs=[output_rgb, output_depth, output_3d],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)