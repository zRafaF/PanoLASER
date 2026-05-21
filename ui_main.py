import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
import os
import torch
from PIL import Image
import copy

from inference_engine.utils.masking import get_spherical_valid_mask
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from inference_engine.inference_utils import align_cam_pts_irls
from inference_engine.utils.geometry import unproject_equirectangular_to_points
from pano_wrapper import PanoVGGTExtractor

extractor = PanoVGGTExtractor()

# --- Helper Functions ---
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

# --- Pipeline 1: Single Frame ---
def process_single_frame(input_image_pil, zenith_limit, nadir_limit, target_width, target_height):
    if input_image_pil is None: return None, None, None, None
    input_image_pil = input_image_pil.resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
    input_image = np.array(input_image_pil)
    H, W = input_image.shape[:2]
    
    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    masked_rgb_vis = visualize_polar_mask(input_image, mask)
    
    preds = extractor.process_frame(input_image)
    depth_map = preds["depth"]
    
    # Apply Spherical Projection to fix the Bubble bug
    xyz_points = unproject_equirectangular_to_points(depth_map)
    
    depth_map[~mask] = 0.0
    depth_vis = visualize_depth(depth_map)
    
    pcd = get_o3d_pcd(xyz_points, input_image, mask)
    return Image.fromarray(masked_rgb_vis), Image.fromarray(depth_vis), create_plotly_figure_from_pcd(pcd), save_pcd_to_ply(pcd)


def apply_transform_to_pcd(pcd, R, t, scale):
    """Transforms an Open3D point cloud using the Kabsch outputs."""
    pcd_scaled = copy.deepcopy(pcd)
    points = np.asarray(pcd_scaled.points)
    
    # Scale -> Rotate -> Translate
    points = points * scale
    points = (R @ points.T).T + t
    
    pcd_scaled.points = o3d.utility.Vector3dVector(points)
    return pcd_scaled

def process_sequence(image_files, zenith_limit, nadir_limit, target_width, target_height):
    if not image_files or len(image_files) < 2:
        raise gr.Error("Please upload at least 2 images to test sequence alignment.")
    
    image_files = sorted(image_files, key=lambda x: x.name)
    
    # 1. Preload and resize all images
    frames = []
    masks = []
    for f in image_files:
        img_pil = Image.open(f.name).convert("RGB").resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
        img_np = np.array(img_pil)
        frames.append(img_np)
        
        mask = get_spherical_valid_mask(img_np.shape[0], img_np.shape[1], zenith_deg=zenith_limit, nadir_deg=nadir_limit)
        masks.append(mask)

    global_pcd = o3d.geometry.PointCloud()
    
    # Tracking variables for the overlapping frame
    prev_overlap_pts = None
    prev_overlap_pose = None
    
    # Sliding Window Parameters
    window_size = 2
    overlap = 1
    
    for i in range(0, len(frames) - 1, window_size - overlap):
        # Extract the current window [A, B]
        window_frames = frames[i : i + window_size]
        window_masks = masks[i : i + window_size]
        
        print(f"--- Processing Window {i} ---")
        preds = extractor.process_window(window_frames)
        depths, poses = preds["depths"], preds["poses"]
        
        # Unproject points for the window
        pts_list = [unproject_equirectangular_to_points(d) for d in depths]
        
        if i == 0:
            # First Window: Forms the anchor of the global map
            for j in range(window_size):
                pcd = get_o3d_pcd(pts_list[j], window_frames[j], window_masks[j])
                
                # Transform using the network's native relative camera pose
                T = poses[j]
                pcd.transform(T)
                global_pcd += pcd
                
            # Store the last frame (Frame B) as the overlap anchor for the next window
            prev_overlap_pts = pts_list[-1]
            prev_overlap_pose = poses[-1]
            
        else:
            # Subsequent Windows: Need to stitch Frame B (current) to Frame B (previous)
            curr_overlap_pts = pts_list[0]
            curr_overlap_pose = poses[0]
            mask_torch = torch.from_numpy(window_masks[0])
            
            # 1. IRLS Scale Alignment (On dense points)
            scale_diff = align_cam_pts_irls(
                torch.from_numpy(curr_overlap_pts), 
                torch.from_numpy(prev_overlap_pts), 
                mask_torch
            )
            
            # 2. Kabsch Rigid Alignment (Strictly on the Camera Pose anchors!)
            R, t = register_camera_poses_kabsch(
                np.expand_dims(curr_overlap_pose, axis=0), 
                np.expand_dims(prev_overlap_pose, axis=0), 
                scale=scale_diff
            )
            print(f"[LASER Align] Scale: {scale_diff:.4f}")
            
            # 3. Apply the calculated Extrinsic Registration to the NEW frame (Frame C)
            new_frame_pts = pts_list[1]
            new_frame_pcd = get_o3d_pcd(new_frame_pts, window_frames[1], window_masks[1])
            
            # Transform Frame C into its local window coordinate space
            new_frame_pcd.transform(poses[1])
            
            # Apply the Kabsch Alignment to shift Frame C into the Global space
            new_frame_pcd = apply_transform_to_pcd(new_frame_pcd, R, t, scale_diff)
            
            global_pcd += new_frame_pcd
            
            # Update overlap anchors
            prev_overlap_pts = np.asarray(new_frame_pcd.points).reshape(target_height, target_width, 3)
            # Pose update math omitted for simplicity here, we trust the points for the next IRLS

    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
    
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
                    gr.Markdown("Upload a sequential batch of overlapping panoramic frames. The engine will correct scale drift using IRLS and stitch them into a global map.")
                    input_seq = gr.File(label="Upload Image Sequence", file_count="multiple", file_types=["image"])
                    run_seq_btn = gr.Button("Align & Stitch Sequence", variant="primary")
                    output_3d_seq = gr.Plot(label="Global Stitched Map")
                    download_seq = gr.File(label="💾 Download Global .ply")

    run_single_btn.click(
        fn=process_single_frame,
        inputs=[input_img, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_rgb, output_depth, output_3d_single, download_single],
        api_name=False
    )
    
    run_seq_btn.click(
        fn=process_sequence,
        inputs=[input_seq, zenith_slider, nadir_slider, target_width, target_height],
        outputs=[output_3d_seq, download_seq],
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)