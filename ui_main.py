import gradio as gr
import numpy as np
from PIL import Image
from inference_engine.utils.masking import get_spherical_valid_mask
from inference_engine.utils.visualization import visualize_polar_mask, visualize_depth
from pano_wrapper import PanoVGGTExtractor

# Initialize the model wrapper globally so it stays in GPU memory
extractor = PanoVGGTExtractor()

def process_pipeline(input_image_pil, zenith_limit, nadir_limit):
    # Safe fallback if no image is uploaded
    if input_image_pil is None:
        return None, None
    
    # 1. Convert the stable PIL Image into a NumPy array for our math
    input_image = np.array(input_image_pil)
    
    H, W = input_image.shape[:2]
    
    # 2. Generate the Exclusion Mask
    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    
    # 3. Visualize the Mask applied to the RGB image
    masked_rgb_vis = visualize_polar_mask(input_image, mask)
    
    # 4. Run PanoVGGT Inference
    predictions = extractor.process_frame(input_image)
    
    # 5. Apply mask to the Depth Map (zeroing out the poles)
    depth_map = predictions["depth"]
    depth_map[~mask] = 0.0  # Zero out invalid depth
    
    # 6. Visualize Depth
    depth_vis = visualize_depth(depth_map)
    
    # 7. The Fix: Convert the NumPy arrays back to PIL Images before returning.
    # This completely bypasses the Gradio 5.x JSON Schema bug.
    return Image.fromarray(masked_rgb_vis), Image.fromarray(depth_vis)

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="PanoLASER Streaming Engine") as demo:
    gr.Markdown("# 🌐 PanoLASER: Spherical Masking & Inference Sandbox")
    gr.Markdown("Test the polar exclusion masks and view PanoVGGT's geometry extraction before streaming.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Set type="pil" for maximum UI stability
            input_img = gr.Image(label="Input 360° Image (Equirectangular)", type="pil")
            
            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=75, step=1, label="Zenith Limit (Top: +Degrees)")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=-65, step=1, label="Nadir Limit (Bottom: -Degrees)")
            
            run_btn = gr.Button("Process Frame", variant="primary")
            
        with gr.Column(scale=2):
            # Set type="pil" for outputs as well
            output_rgb = gr.Image(label="Masked Input (Excluded zones tinted red)", type="pil")
            output_depth = gr.Image(label="Masked Depth Map Prediction", type="pil")

    run_btn.click(
        fn=process_pipeline,
        inputs=[input_img, zenith_slider, nadir_slider],
        outputs=[output_rgb, output_depth]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)