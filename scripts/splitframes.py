import cv2
import os

def extract_mask_and_compress_frames(
    video_path="video.mp4",
    output_folder="examples",
    nadir_degrees=-60,
    zenith_degrees=75,
    mode="black",
    target_fps=5,
    jpeg_quality=80,
    output_width=1036,
    output_height=518
):
    """
    Extracts, downsamples, masks, and compresses frames from a 360 video.

    :param nadir_degrees:  Latitude of the nadir cutoff in degrees (-90 to 0).
                           Pixels below this latitude are masked. Default: -60°
    :param zenith_degrees: Latitude of the zenith cutoff in degrees (0 to 90).
                           Pixels above this latitude are masked. Default: 75°
    :param jpeg_quality:   1-100 (Higher means better quality). 80 is optimal.
    :param output_width:   Target width in pixels after scaling.
    :param output_height:  Target height in pixels after scaling.
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

    print(f"Processing: {target_fps} FPS | Quality: {jpeg_quality}% | "
          f"Output: {output_width}x{output_height}px | "
          f"Zenith: {zenith_degrees}° | Nadir: {nadir_degrees}°")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if source_frame_count == int(next_frame_to_save):
            h, w = frame.shape[:2]

            # --- Equirectangular degree-to-pixel conversion ---
            # In an equirectangular image:
            #   latitude  +90° → row 0   (top)
            #   latitude   0°  → row h/2 (middle)
            #   latitude  -90° → row h   (bottom)
            # Formula: row = (90 - lat_degrees) / 180 * h

            # 1. Apply zenith mask (top of frame)
            zenith_row = int((90 - zenith_degrees) / 180 * h)
            if mode == "black":
                frame[0:zenith_row, 0:w] = 0
            elif mode == "blur":
                region = frame[0:zenith_row, 0:w]
                frame[0:zenith_row, 0:w] = cv2.GaussianBlur(region, (99, 99), 0)

            # 2. Apply nadir mask (bottom of frame)
            nadir_row = int((90 - nadir_degrees) / 180 * h)
            if mode == "black":
                frame[nadir_row:h, 0:w] = 0
            elif mode == "blur":
                region = frame[nadir_row:h, 0:w]
                frame[nadir_row:h, 0:w] = cv2.GaussianBlur(region, (99, 99), 0)

            # 3. Resize to exact pixel dimensions
            frame = cv2.resize(
                frame, (output_width, output_height),
                interpolation=cv2.INTER_AREA
            )

            # 4. Save with JPEG compression
            frame_name = f"frame_{str(saved_frame_count).zfill(4)}.jpg"
            output_path = os.path.join(output_folder, frame_name)
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
        nadir_degrees=-60,
        zenith_degrees=75,
        mode="black",
        target_fps=5,
        jpeg_quality=100,
        output_width=1036,
        output_height=518
    )