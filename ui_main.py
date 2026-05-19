import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
import os
from PIL import Image

from inference_engine.utils.masking import get_spherical_valid_mask
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from pano_wrapper import PanoVGGTExtractor

# Initialize the model wrapper globally
extractor = PanoVGGTExtractor()

def create_point_cloud_ply(xyz_points, rgb_image, mask):
    """Creates the full-density PLY file for local visualization."""
    valid_mask_flat = mask.flatten()
    points_flat = xyz_points.reshape(-1, 3)[valid_mask_flat]
    colors_flat = rgb_image.reshape(-1, 3)[valid_mask_flat] / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_flat)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)
    
    temp_dir = tempfile.mkdtemp()
    ply_path = os.path.join(temp_dir, "reconstruction_dense.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    return ply_path

def create_plotly_figure(xyz_points, rgb_image, mask, max_points=150000):
    """Creates a subsampled Plotly figure for robust in-browser rendering."""
    valid_mask_flat = mask.flatten()
    points_flat = xyz_points.reshape(-1, 3)[valid_mask_flat]
    colors_rgb = rgb_image.reshape(-1, 3)[valid_mask_flat]
    
    if len(points_flat) > max_points:
        indices = np.random.choice(len(points_flat), max_points, replace=False)
        points_flat = points_flat[indices]
        colors_rgb = colors_rgb[indices]
        
    colors_str = [f"rgb({r},{g},{b})" for r, g, b in colors_rgb]
    
    fig = go.Figure(
        data=[go.Scatter3d(
            x=points_flat[:, 0],      
            y=points_flat[:, 2],      
            z=-points_flat[:, 1],     
            mode='markers',
            marker=dict(size=1.5, color=colors_str, opacity=1.0)
        )]
    )
    
    fig.update_layout(
        scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#111111",
    )
    return fig

def enforce_resolution(w, h, step, link, trigger):
    step = max(1, int(step))
    if link:
        if trigger == 'w': h = w / 2.0
        elif trigger == 'h': w = h * 2.0
            
    w_snap = int(round(w / step) * step)
    h_snap = int(round(h / step) * step)
    actual_ratio = w_snap / h_snap if h_snap > 0 else 0
    error = abs(2.0 - actual_ratio)
    
    msg = f"📐 **Processing Dimensions:** {w_snap} $\\times$ {h_snap} | **Target Ratio:** 2.0 | **Actual:** {actual_ratio:.4f} | **Error:** {error:.4f}"
    return w_snap, h_snap, msg

def process_pipeline(input_image_pil, zenith_limit, nadir_limit, target_width, target_height):
    if input_image_pil is None:
        return None, None, None, None
    
    input_image_pil = input_image_pil.resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
    input_image = np.array(input_image_pil)
    H, W = input_image.shape[:2]
    
    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    masked_rgb_vis = visualize_polar_mask(input_image, mask)
    
    predictions = extractor.process_frame(input_image)
    depth_map = predictions["depth"]
    xyz_points = predictions["points"]  # <-- Using the Native 3D Output
    
    depth_map[~mask] = 0.0
    depth_vis = visualize_depth(depth_map)
    
    plotly_fig = create_plotly_figure(xyz_points, input_image, mask)
    ply_file_path = create_point_cloud_ply(xyz_points, input_image, mask)
    
    return Image.fromarray(masked_rgb_vis), Image.fromarray(depth_vis), plotly_fig, ply_file_path


# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="PanoLASER Streaming Engine") as demo:
    gr.Markdown("# 🌐 PanoLASER: Spherical Masking & Inference Sandbox")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Input 360° Image", type="pil")
            
            gr.Markdown("### Processing Resolution")
            with gr.Row():
                step_size = gr.Number(value=14, label="Step Size (Patch Size)")
                link_ratio = gr.Checkbox(value=True, label="Link Aspect Ratio (2:1)")
                
            target_width = gr.Slider(minimum=224, maximum=4096, value=1036, step=1, label="Target Width")
            target_height = gr.Slider(minimum=112, maximum=2048, value=518, step=1, label="Target Height")
            ratio_info = gr.Markdown("📐 **Processing Dimensions:** 1036 $\\times$ 518 | **Target Ratio:** 2.0 | **Actual:** 2.0000 | **Error:** 0.0000")
            
            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-60, step=1, label="Nadir Limit")
            
            run_btn = gr.Button("Process Frame & Generate 3D", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("2D Masking Analysis"):
                    output_rgb = gr.Image(label="Masked Input", type="pil")
                    output_depth = gr.Image(label="Masked Depth Map Prediction", type="pil")
                
                with gr.Tab("3D Web Visualizer (Subsampled)"):
                    output_3d = gr.Plot(label="Interactive Point Cloud (WebGL)")
            
            # Moved outside the tabs so it is always visible
            download_ply = gr.File(label="💾 Download Full Dense Point Cloud (.ply)")

    # FIX: Changed .change() to .release() to prevent UI freezing
    target_width.release(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info]
    )
    target_height.release(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'h'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info]
    )
    link_ratio.change(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info]
    )
    step_size.release(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info]
    )

    run_btn.click(
        fn=process_pipeline,
        inputs=[input_img, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_rgb, output_depth, output_3d, download_ply],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)