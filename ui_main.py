import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
import os
from PIL import Image

# Import our new architecture
from pano_wrapper import PanoVGGTExtractor
from inference_engine.vanilla_engine import PanoVanillaEngine
from inference_engine.streaming_window_engine import PanoStreamingEngine
from inference_engine.utils.masking import get_spherical_valid_mask

# Initialize the stack
print("Initializing Architecture Stack...")
base_model_wrapper = PanoVGGTExtractor()
vanilla_engine = PanoVanillaEngine(base_model_wrapper.model)
streaming_engine = PanoStreamingEngine(vanilla_engine, window_size=2, overlap=1)

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

    # The magic call to the new architecture
    global_pcd = streaming_engine.process_sequence(frames, masks)
    
    return create_plotly_figure_from_pcd(global_pcd), save_pcd_to_ply(global_pcd, "global_stitched_map")

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="PanoLASER Streaming Engine") as demo:
    gr.Markdown("# 🌐 PanoLASER: Alignment & Streaming Sandbox")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Processing Controls")
            target_width = gr.Slider(minimum=224, maximum=4096, value=1036, step=14, label="Target Width")
            target_height = gr.Slider(minimum=112, maximum=2048, value=518, step=14, label="Target Height")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-60, step=1, label="Nadir Limit")
            
        with gr.Column(scale=2):
            input_seq = gr.File(label="Upload Image Sequence", file_count="multiple", file_types=["image"])
            run_seq_btn = gr.Button("Align & Stitch Sequence", variant="primary")
            output_3d_seq = gr.Plot(label="Global Stitched Map")
            download_seq = gr.File(label="💾 Download Global .ply")

    run_seq_btn.click(
        fn=process_sequence_ui,
        inputs=[input_seq, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_3d_seq, download_seq],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)