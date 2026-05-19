import gradio as gr
import numpy as np
import cv2
import os
from inference_engine.utils.masking import get_spherical_valid_mask, apply_mask_to_tensor
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from pano_wrapper import PanoVGGTExtractor

# Initialize the model wrapper globally so it stays in GPU memory
extractor = PanoVGGTExtractor()

def process_pipeline(image_path, zenith_limit, nadir_limit):
    # Safe fallback if no image is uploaded
    if not image_path or not os.path.exists(image_path):
        return None, None
    
    # Manually load the image via OpenCV to bypass Gradio's numpy schema bug
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
        
    # OpenCV loads in BGR, Gradio and Models expect RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    H, W = input_image.shape[:2]
    
    # 1. Generate the Exclusion Mask
    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    
    # 2. Visualize the Mask applied to the RGB image
    masked_rgb_vis = visualize_polar_mask(input_image, mask)
    
    # 3. Run PanoVGGT Inference
    predictions = extractor.process_frame(input_image)
    
    # 4. Apply mask to the Depth Map (zeroing out the poles)
    depth_map = predictions["depth"]
    depth_map[~mask] = 0.0  # Zero out invalid depth
    
    # 5. Visualize Depth
    depth_vis = visualize_depth(depth_map)
    
    return masked_rgb_vis, depth_vis

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="PanoLASER Streaming Engine") as demo:
    gr.Markdown("# 🌐 PanoLASER: Spherical Masking & Inference Sandbox")
    gr.Markdown("Test the polar exclusion masks and view PanoVGGT's geometry extraction before streaming.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Input 360° Image (Equirectangular)", type="filepath")
            
            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit (Top: +Degrees)")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-65, step=1, label="Nadir Limit (Bottom: -Degrees)")
            
            run_btn = gr.Button("Process Frame", variant="primary")
            
        with gr.Column(scale=2):
            # FIX: Removed type="numpy" from the output components
            output_rgb = gr.Image(label="Masked Input (Excluded zones tinted red)")
            output_depth = gr.Image(label="Masked Depth Map Prediction")

    run_btn.click(
        fn=process_pipeline,
        inputs=[input_img, zenith_slider, nadir_slider],
        outputs=[output_rgb, output_depth]
    )

if __name__ == "__main__":
    # share=True resolves the 'localhost not accessible' crash on cloud VMs
    # It will print a public *.gradio.live URL in the terminal for you to access.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)