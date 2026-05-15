import cv2
import os

def extract_mask_and_compress_frames(
    video_path="video.mp4", 
    output_folder="examples", 
    nadir_ratio=0.165, 
    mode="black", 
    target_fps=5,
    jpeg_quality=80,
    scale_factor=1.0
):
    """
    Extracts, downsamples, masks, and compresses frames from a 360 video.
    
    :param jpeg_quality: 1-100 (Higher means better quality, lower means smaller file size). 80 is optimal.
    :param scale_factor: Float multiplier for resolution (0.5 halves the width/height, reducing file size by ~75%).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: '{output_folder}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    frame_step = source_fps / target_fps
    next_frame_to_save = 0.0
    source_frame_count = 0
    saved_frame_count = 0
    
    print(f"Processing: {target_fps} FPS | Quality: {jpeg_quality}% | Scale: {scale_factor}x")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if source_frame_count == int(next_frame_to_save):
            h, w, _ = frame.shape
            
            # 1. Apply the nadir patch first while at native resolution
            nadir_start_y = int(h * (1 - nadir_ratio))
            if mode == "black":
                frame[nadir_start_y:h, 0:w] = 0
            elif mode == "blur":
                nadir_region = frame[nadir_start_y:h, 0:w]
                blurred_nadir = cv2.GaussianBlur(nadir_region, (99, 99), 0)
                frame[nadir_start_y:h, 0:w] = blurred_nadir

            # 2. Downscale resolution if scale_factor is less than 1.0
            if scale_factor != 1.0:
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                # INTER_AREA is ideal for shrinking images without artifacts
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 3. Save the frame with custom JPEG compression quality
            frame_name = f"frame_{str(saved_frame_count).zfill(4)}.jpg"
            output_path = os.path.join(output_folder, frame_name)
            
            # OpenCV requires the quality parameter to be passed as an integer list pair
            cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            
            saved_frame_count += 1
            next_frame_to_save += frame_step
        
        source_frame_count += 1

    cap.release()
    print("---")
    print(f"Finished! Saved {saved_frame_count} compressed frames to '{output_folder}/'.")

if __name__ == "__main__":
    extract_mask_and_compress_frames(
        video_path="video.mp4", 
        output_folder="examples", 
        nadir_ratio=0.165, 
        mode="black", 
        target_fps=5,
        
        # --- NEW OPTIMIZATION SETTINGS ---
        jpeg_quality=90,   # Dropping from 95 to 80 massively shrinks file size safely
        scale_factor=0.9   # 0.5 cuts resolution in half. Change to 1.0 to keep original size.
    )