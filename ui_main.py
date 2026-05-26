import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
import os
from PIL import Image

from pano_wrapper import PanoVGGTExtractor
from inference_engine.vanilla_engine import PanoVanillaEngine
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC

from inference_engine.utils.masking import get_spherical_valid_mask
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from inference_engine.utils.geometry import unproject_equirectangular_to_points

print("Initializing Architecture Stack...")
base_model_wrapper = PanoVGGTExtractor()
vanilla_engine = PanoVanillaEngine(base_model_wrapper.model)
streaming_engine = StreamingWindowEngineLC(vanilla_engine, window_size=16, overlap=4)

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

def create_plotly_figure_with_trajectory(pcd, trajectory, lc_edges):
    """Generates a black-canvas Plotly scatter plot with cyan paths and red loop closure lines."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Decimate dense point clouds for smoother web browser rendering
    max_ui_points = 600000
    if len(points) > max_ui_points:
        idx = np.random.choice(len(points), max_ui_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    fig = go.Figure()
    
    # Trace 1: Dense Reconstructed Geometry
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=1.2, color=colors),
        name='Environment Map'
    ))
    
    # Trace 2: Cyan Odometry Trajectory Path
    if trajectory is not None and len(trajectory) > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode='lines+markers',
            name='Submap Trajectory',
            line=dict(color='cyan', width=5),
            marker=dict(size=5, color='orange', symbol='circle')
        ))
        
    # Trace 3: Red Disconnected Loop-Closure Segments
    if lc_edges is not None and len(lc_edges) > 0:
        lc_x, lc_y, lc_z = [], [], []
        for p1, p2 in lc_edges:
            lc_x.extend([p1[0], p2[0], None])
            lc_y.extend([p1[1], p2[1], None])
            lc_z.extend([p1[2], p2[2], None])
            
        fig.add_trace(go.Scatter3d(
            x=lc_x, y=lc_y, z=lc_z,
            mode='lines',
            name='Loop Constraints',
            line=dict(color='rgb(255, 30, 30)', width=6, dash='dash')
        ))
        
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            bgcolor='black',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='gray'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='gray'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='gray')
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0.02, y=0.98, font=dict(color="white"), bgcolor="rgba(0,0,0,0.5)")
    )
    return fig

def process_single_frame(input_img, zenith_deg, nadir_deg, target_w, target_h):
    if input_img is None:
        return None, None, None, None
    img_pil = Image.fromarray(input_img).resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil)
    preds = vanilla_engine([img_np])
    xyz_points = preds["points"][0]
    mask = get_spherical_valid_mask(img_np.shape[0], img_np.shape[1], zenith_deg, nadir_deg)
    pcd = get_o3d_pcd(xyz_points, img_np, mask)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz_points[mask.astype(bool), 0],
        y=xyz_points[mask.astype(bool), 1],
        z=xyz_points[mask.astype(bool), 2],
        mode='markers', marker=dict(size=1.5, color=img_np[mask.astype(bool)]/255.0)
    )])
    fig.update_layout(scene=dict(aspectmode='data', bgcolor='black'), margin=dict(l=0, r=0, b=0, t=0))
    return img_np, visualize_depth(xyz_points, mask), fig, save_pcd_to_ply(pcd, "single_frame")

def process_sequence_ui(mode, upload_seq, local_dir, decimation, zenith, nadir, tw, th, ws, ov):
    frames, masks = [], []
    
    if mode == "Upload File Sequence" and upload_seq:
        files = sorted(upload_seq, key=lambda x: x.name)[::int(decimation)]
        for f in files:
            img = Image.open(f.name).convert("RGB").resize((int(tw), int(th)), Image.Resampling.LANCZOS)
            frames.append(np.array(img))
    elif mode == "Local Storage Directory" and local_dir:
        import glob
        paths = sorted(glob.glob(os.path.join(local_dir, "*.jpg")) + glob.glob(os.path.join(local_dir, "*.png")))[::int(decimation)]
        for p in paths:
            img = Image.open(p).convert("RGB").resize((int(tw), int(th)), Image.Resampling.LANCZOS)
            frames.append(np.array(img))
            
    if len(frames) < 2:
        raise gr.Error("Sequence loader found less than 2 valid images.")

    for f in frames:
        masks.append(get_spherical_valid_mask(f.shape[0], f.shape[1], zenith, nadir))

    streaming_engine.window_size = int(ws)
    streaming_engine.overlap = int(ov)

    # Unpack advanced structural trajectory variables
    global_pcd, trajectory, lc_edges = streaming_engine.process_sequence(frames, masks)
    
    fig = create_plotly_figure_with_trajectory(global_pcd, trajectory, lc_edges)
    return fig, save_pcd_to_ply(global_pcd, "sequence_map_output")

def enforce_resolution(w, h, step, link, trigger_axis):
    step = int(step)
    if trigger_axis == 'w':
        w = int(np.round(w / step) * step)
        if link: h = int(w // 2)
    else:
        h = int(np.round(h / step) * step)
        if link: w = int(h * 2)
    return w, h, f"Linked Aspect Check: {w}x{h} (Multiple of {step})"

def toggle_input_mode(choice):
    return gr.update(visible=(choice == "Upload File Sequence")), gr.update(visible=(choice == "Local Storage Directory"))

# --- Gradio UI Layout Building Block ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate")) as demo:
    gr.Markdown("# PanoLASER: 360° Panoramic Submap Factor-Graph Engine")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Processing Dimensions")
            target_width = gr.Number(value=1036, label="Width")
            target_height = gr.Number(value=518, label="Height")
            step_size = gr.Dropdown(choices=[14, 28, 56], value=14, label="Resolution Quant Step")
            link_ratio = gr.Checkbox(value=True, label="Lock 2:1 Equirectangular Ratio")
            ratio_info = gr.Markdown("Ratio Status: Valid")
            
            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Cutoff")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-60, step=1, label="Nadir Cutoff")
            
            gr.Markdown("### SLAM Configurations")
            window_size_slider = gr.Slider(minimum=4, maximum=32, value=16, step=1, label="Batch Window Size")
            overlap_slider = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Overlap Frames")

        with gr.Column(scale=2):
            with gr.Tab("Stream Mapping Sequence"):
                input_mode = gr.Radio(["Upload File Sequence", "Local Storage Directory"], value="Upload File Sequence", label="Data Source")
                input_seq = gr.File(file_count="multiple", label="Sequence Upload Container")
                local_dir_input = gr.Textbox(placeholder="/path/to/frames", label="Local Host Folder PATH", visible=False)
                decimation_input = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Frame Decimation Rate")
                
                run_seq_btn = gr.Button("Initialize Factor Graph SLAM Run", variant="primary")
                output_3d_seq = go.Scatter3d() # Will map cleanly to gr.Plotly output container
                output_3d_seq = gr.Plotly(label="Global Metric 3D Model", responsive=True)
                download_seq = gr.File(label="Download Full Map Model (.ply)")

            with gr.Tab("Single Frame Extraction Diagnoses"):
                input_img = gr.Image(label="Source Panoramic View Frame")
                run_single_btn = gr.Button("Extract Geometry Assets", variant="secondary")
                output_rgb = gr.Image(label="Segmented Image")
                output_depth = gr.Image(label="Estimated Geometric Depth map")
                output_3d_single = gr.Plotly(label="Local 3D Map")
                download_single = gr.File(label="Download Mesh (.ply)")

    # Event handlers link setup
    target_width.change(fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height, ratio_info])
    target_height.change(fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'h'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height, ratio_info])
    input_mode.change(fn=toggle_input_mode, inputs=input_mode, outputs=[input_seq, local_dir_input])
    
    run_single_btn.click(fn=process_single_frame, inputs=[input_img, zenith_slider, nadir_slider, target_width, target_height], outputs=[output_rgb, output_depth, output_3d_single, download_single])
    run_seq_btn.click(fn=process_sequence_ui, inputs=[input_mode, input_seq, local_dir_input, decimation_input, zenith_slider, nadir_slider, target_width, target_height, window_size_slider, overlap_slider], outputs=[output_3d_seq, download_seq])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)