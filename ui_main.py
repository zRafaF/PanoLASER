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


def save_mesh_to_glb(mesh, prefix="reconstruction"):
    temp_dir = tempfile.mkdtemp()
    glb_path = os.path.join(temp_dir, f"{prefix}.glb")
    o3d.io.write_triangle_mesh(glb_path, mesh)
    return glb_path


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
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#111111"
    )
    return fig


def create_plotly_figure_with_trajectory(pcd, trajectory, lc_edges, max_points=150000):
    # 1. Base Point Cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    colors_str = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 2], z=-points[:, 1],
        mode='markers',
        marker=dict(size=1.5, color=colors_str, opacity=1.0),
        name='Geometry'
    ))
    
    # 2. Camera Trajectory (Cyan Line with Nodes)
    if trajectory is not None and len(trajectory) > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 2], z=-trajectory[:, 1],
            mode='lines+markers',
            name='Camera Trajectory',
            line=dict(color='cyan', width=4),
            marker=dict(size=4, color='orange')
        ))
        
    # 3. Loop Closures (Red Dashed Lines)
    if lc_edges is not None and len(lc_edges) > 0:
        lc_x, lc_y, lc_z = [], [], []
        for p1, p2 in lc_edges:
            # Match the point cloud axis mapping (x=0, y=2, z=-1)
            lc_x.extend([p1[0], p2[0], None])
            lc_y.extend([p1[2], p2[2], None])
            lc_z.extend([-p1[1], -p2[1], None])
            
        fig.add_trace(go.Scatter3d(
            x=lc_x, y=lc_y, z=lc_z,
            mode='lines',
            name='Loop Closures',
            line=dict(color='red', width=6, dash='dash')
        ))
        
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="#111111",
        legend=dict(x=0.02, y=0.98, font=dict(color="white"))
    )
    return fig


def enforce_resolution(w, h, step, link, trigger):
    step = max(1, int(step))
    if link:
        if trigger == 'w':
            h = w / 2.0
        elif trigger == 'h':
            w = h * 2.0
    w_snap = int(round(w / step) * step)
    h_snap = int(round(h / step) * step)
    actual_ratio = w_snap / h_snap if h_snap > 0 else 0
    error = abs(2.0 - actual_ratio)
    msg = (
        f"📐 **Processing Dimensions:** {w_snap} $\\times$ {h_snap} | "
        f"**Target Ratio:** 2.0 | **Actual:** {actual_ratio:.4f} | **Error:** {error:.4f}"
    )
    return w_snap, h_snap, msg


def process_single_frame(input_image_pil, zenith_limit, nadir_limit, target_width, target_height):
    if input_image_pil is None:
        return None, None, None, None
    input_image_pil = input_image_pil.resize(
        (int(target_width), int(target_height)), Image.Resampling.LANCZOS
    )
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
    return (
        Image.fromarray(masked_rgb_vis),
        Image.fromarray(depth_vis),
        create_plotly_figure_from_pcd(pcd),
        save_pcd_to_ply(pcd, "single_frame"),
    )


# --- File Fetching & Decimation Logic ---
def get_file_list(input_mode, uploaded_files, local_dir, decimation):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = []

    if input_mode == "Upload Files":
        if uploaded_files:
            files = sorted([f.name for f in uploaded_files])
    else:
        if local_dir and os.path.isdir(local_dir):
            raw_files = os.listdir(local_dir)
            files = sorted([
                os.path.join(local_dir, f)
                for f in raw_files
                if f.lower().endswith(valid_exts)
            ])

    step = max(1, int(decimation) + 1)
    return files[::step]


def check_files_ui(input_mode, uploaded_files, local_dir, decimation):
    files = get_file_list(input_mode, uploaded_files, local_dir, decimation)
    if not files:
        return "⚠️ No valid images found or provided."

    names = [os.path.basename(f) for f in files]
    out = f"✅ Total files to process: {len(names)}\n\n"
    out += "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
    return out


def process_sequence_ui(
    input_mode, uploaded_files, local_dir, decimation,
    zenith_limit, nadir_limit,
    target_width, target_height,
    window_size, overlap,
):
    file_paths = get_file_list(input_mode, uploaded_files, local_dir, decimation)

    if not file_paths or len(file_paths) < 2:
        raise gr.Error(
            "Please provide at least 2 valid images "
            "(upload files or point to a local directory with images)."
        )

    frames, masks = [], []
    for path in file_paths:
        img_pil = (
            Image.open(path)
            .convert("RGB")
            .resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
        )
        img_np = np.array(img_pil)
        frames.append(img_np)
        mask = get_spherical_valid_mask(
            img_np.shape[0], img_np.shape[1],
            zenith_deg=zenith_limit, nadir_deg=nadir_limit,
        )
        masks.append(mask)

    streaming_engine.window_size = int(window_size)
    streaming_engine.overlap = int(overlap)

    # --- FIX: Loop over the generator for live streaming ---
    for mesh, global_pcd, trajectory, lc_edges in streaming_engine.process_sequence(frames, masks):
        
        # Render the live point cloud and trajectory
        fig = create_plotly_figure_with_trajectory(global_pcd, trajectory, lc_edges)
        pcd_path = save_pcd_to_ply(global_pcd, "live_stitched_map")
        
        # If the mesh is None (during the stream), use gr.skip() to tell Gradio not to update those UI elements yet
        if mesh is None:
            yield fig, pcd_path, gr.skip(), gr.skip()
        else:
            # The sequence has finished, export and display the final GLB mesh
            mesh_path = save_mesh_to_glb(mesh, "final_stitched_mesh")
            yield fig, pcd_path, mesh_path, mesh_path


def toggle_input_mode(mode):
    if mode == "Upload Files":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


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
            ratio_info = gr.Markdown(
                "📐 **Processing Dimensions:** 1036 $\\times$ 518 | "
                "**Target Ratio:** 2.0 | **Actual:** 2.0000 | **Error:** 0.0000"
            )

            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-60, step=1, label="Nadir Limit")

            gr.Markdown("### Submap Configuration (SLAM)")
            window_size_slider = gr.Slider(minimum=3, maximum=32, value=16, step=1, label="Submap Window Size (Frames Batch)")
            overlap_slider = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Submap Frame Overlap")

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
                    gr.Markdown(
                        "Select your source type. You can either drag & drop files, "
                        "or provide a local server directory path."
                    )

                    input_mode = gr.Radio(
                        choices=["Upload Files", "Local Directory Path"],
                        value="Upload Files",
                        label="Input Mode",
                    )

                    input_seq = gr.File(
                        label="Upload Image Sequence (Drag & Drop)",
                        file_count="multiple",
                        file_types=["image"],
                        visible=True,
                    )
                    local_dir_input = gr.Textbox(
                        label="Absolute Local Directory Path (e.g., /app/data/sequence1)",
                        visible=False,
                    )

                    with gr.Row():
                        decimation_input = gr.Number(
                            value=0, label="Decimation (Skip N files)", precision=0,
                            info="0 = keep all. 1 = skip every 1 file (take 1/2), 2 = skip 2 files, etc."
                        )
                        check_files_btn = gr.Button("Check Files & Preview Queue")

                    checked_files_output = gr.Textbox(
                        label="Files to be Processed", interactive=False, lines=5
                    )

                    run_seq_btn = gr.Button("Align & Stitch Sequence", variant="primary")
                    
                    with gr.Tabs():
                        with gr.Tab("Point Cloud Viewer"):
                            output_3d_seq = gr.Plot(label="Global Stitched Map")
                            download_seq = gr.File(label="💾 Download Global .ply")
                        with gr.Tab("High-Res Mesh"):
                            output_mesh = gr.Model3D(label="High-Res Poisson Mesh")
                            download_mesh = gr.File(label="💾 Download Global .glb")

    # --- Wire-ups ---
    target_width.release(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info],
    )
    target_height.release(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'h'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info],
    )
    link_ratio.change(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info],
    )
    step_size.change(
        fn=lambda w, h, s, l: enforce_resolution(w, h, s, l, 'w'),
        inputs=[target_width, target_height, step_size, link_ratio],
        outputs=[target_width, target_height, ratio_info],
    )

    input_mode.change(
        fn=toggle_input_mode,
        inputs=input_mode,
        outputs=[input_seq, local_dir_input],
    )

    check_files_btn.click(
        fn=check_files_ui,
        inputs=[input_mode, input_seq, local_dir_input, decimation_input],
        outputs=[checked_files_output],
        api_name=False,
    )

    run_single_btn.click(
        fn=process_single_frame,
        inputs=[input_img, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_rgb, output_depth, output_3d_single, download_single],
        api_name=False,
    )

    run_seq_btn.click(
        fn=process_sequence_ui,
        inputs=[
            input_mode, input_seq, local_dir_input, decimation_input,
            zenith_slider, nadir_slider,
            target_width, target_height,
            window_size_slider, overlap_slider,
        ],
        outputs=[output_3d_seq, download_seq, output_mesh, download_mesh],
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)