import cv2
import os

def extract_and_mask_frames(video_path="video.mp4", output_folder="examples", nadir_ratio=0.15, mode="black"):
    """
    Extracts frames and masks out the nadir (bottom) of a 360 video.
    
    :param nadir_ratio: Decimal percentage of the bottom to mask (0.15 = bottom 15% of the frame)
    :param mode: 'black' to paint it solid black, 'blur' to heavily blur it
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: '{output_folder}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return

    frame_count = 0
    print(f"Processing frames using '{mode}' mask mode...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get frame dimensions (Height, Width)
        h, w, _ = frame.shape
        
        # Calculate where the nadir patch starts from the top
        nadir_start_y = int(h * (1 - nadir_ratio))
        
        # Extract the bottom region where you are visible
        nadir_region = frame[nadir_start_y:h, 0:w]
        
        if mode == "black":
            # Option 1: Complete black out (Sets all pixels in the region to 0)
            frame[nadir_start_y:h, 0:w] = 0
            
        elif mode == "blur":
            # Option 2: Heavy Gaussian Blur
            # Kernel size must be odd numbers. Larger numbers = stronger blur.
            blurred_nadir = cv2.GaussianBlur(nadir_region, (99, 99), 0)
            frame[nadir_start_y:h, 0:w] = blurred_nadir

        # Save the modified frame
        frame_name = f"frame_{str(frame_count).zfill(4)}.jpg"
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, frame)
        frame_count += 1

    cap.release()
    print(f"Done! Processed {frame_count} frames into '{output_folder}/'.")

if __name__ == "__main__":
    # Change mode="black" to mode="blur" if you prefer a blurred look instead
    # Adjust nadir_ratio (e.g., 0.10 for less coverage, 0.20 for more coverage)
    extract_and_mask_frames(video_path="video.mp4", output_folder="examples", nadir_ratio=0.15, mode="black")