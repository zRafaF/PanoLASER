import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
import os
from PIL import Image

# Import our architecture
from pano_wrapper import PanoVGGTExtractor
from inference_engine.vanilla_engine import PanoVanillaEngine
from inference_engine.streaming_window_engine import PanoStreamingEngine

from inference_engine.utils.masking import get_spherical_valid_mask
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from inference_engine.utils.geometry import unproject_equirectangular_to_points

print("Initializing Architecture Stack...")
base_model_wrapper = PanoVGGTExtractor()
vanilla_engine = PanoVanillaEngine(base_model_wrapper.model)
streaming_engine = PanoStreamingEngine(vanilla_engine, window_size=2, overlap=1)

def get_o3d_pcd(xyz_points, rgb_image, mask):
    valid_mask = mask.astype(bool)
    points_filtered = xyz_points[valid_mask]
    colors_filtered = rgb_image[valid_mask] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    return pcd

def save_pcd_to_ply(pcd, prefix="reconstruction"):
    temp_dir = tempfile.mkdtemp()
    ply_path = os.path.join(temp_dir, f"{prefix}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    return ply_path

def create_plotly_figure_from_pcd(pcd, max_points=150000):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]
        
    colors_str = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 2], z=-points[:, 1],
        mode='markers', marker=dict(size=1.5, color=colors_str, opacity=1.0)
    )])
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="#111111")
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

def process_single_frame(input_image_pil, zenith_limit, nadir_limit, target_width, target_height):
    if input_image_pil is None: return None, None, None, None
    input_image_pil = input_image_pil.resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
    input_image = np.array(input_image_pil)
    H, W = input_image.shape[:2]
    
    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    masked_rgb_vis = visualize_polar_mask(input_image, mask)
    
    preds = base_model_wrapper.process_frame(input_image)
    depth_map = preds["depth"]
    
    xyz_points = unproject_equirectangular_to_points(np.squeeze(depth_map))
    depth_map[~mask] = 0.0
    depth_vis = visualize_depth(depth_map)
    
    pcd = get_o3d_pcd(xyz_points, input_image, mask)
    return Image.fromarray(masked_rgb_vis), Image.fromarray(depth_vis), create_plotly_figure_from_pcd(pcd), save_pcd_to_ply(pcd, "single_frame")

def process_sequence_ui(image_files, zenith_limit, nadir_limit, target_width, target_height):
    if not image_files or len(image_files) < 2:
        raise gr.Error("Please upload at least 2 images.")
    
    image_files = sorted(image_files, key=lambda x: x.name)
    
    frames = []
    masks = []
    for f in image_files:
        img_pil = Image.open(f.name).convert("RGB").resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
        img_np = np.array(img_pil)
        frames.append(img_np)
        mask = get_spherical_valid_mask(img_np.shape[0], img_np.shape[1], zenith_deg=zenith_limit, nadir_deg=nadir_limit)
        masks.append(mask)

    global_pcd = streaming_engine.process_sequence(frames, masks)
    return create_plotly_figure_from_pcd(global_pcd), save_pcd_to_ply(global_pcd, "global_stitched_map")

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="PanoLASER Streaming Engine") as demo:
    gr.Markdown("# 🌐 PanoLASER: Alignment & Streaming Sandbox")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Processing Controls")
            with gr.Row():
                step_size = gr.Number(value=14, label="Step Size")
                link_ratio = gr.Checkbox(value=True, label="Link Aspect Ratio")
                
            target_width = gr.Slider(minimum=224, maximum=4096, value=1036, step=1, label="Target Width")
            target_height = gr.Slider(minimum=112, maximum=2048, value=518, step=1, label="Target Height")
            ratio_info = gr.Markdown("📐 **Processing Dimensions:** 1036 $\\times$ 518 | **Target Ratio:** 2.0 | **Actual:** 2.0000 | **Error:** 0.0000")
            
            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-60, step=1, label="Nadir Limit")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("1. Single Frame Extract"):
                    input_img = gr.Image(label="Input Single 360° Image", type="pil")
                    run_single_btn = gr.Button("Extract Geometry", variant="primary")
                    with gr.Row():
                        output_rgb = gr.Image(label="Masked Input", type="pil")
                        output_depth = gr.Image(label="Depth Map", type="pil")
                    output_3d_single = gr.Plot(label="Single Frame Point Cloud")
                    download_single = gr.File(label="💾 Download Frame .ply")
                
                with gr.Tab("2. Multi-Frame 4D Stitching"):
                    input_seq = gr.File(label="Upload Image Sequence", file_count="multiple", file_types=["image"])
                    run_seq_btn = gr.Button("Align & Stitch Sequence", variant="primary")
                    output_3d_seq = gr.Plot(label="Global Stitched Map")
                    download_seq = gr.File(label="💾 Download Global .ply")

    # Wire up resolutions
    target_width.release(fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height, ratio_info])
    target_height.release(fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'h'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height, ratio_info])
    link_ratio.change(fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height, ratio_info])
    step_size.change(fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height, ratio_info])

    run_single_btn.click(
        fn=process_single_frame,
        inputs=[input_img, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_rgb, output_depth, output_3d_single, download_single],
        api_name=False
    )
    
    run_seq_btn.click(
        fn=process_sequence_ui,
        inputs=[input_seq, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_3d_seq, download_seq],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)