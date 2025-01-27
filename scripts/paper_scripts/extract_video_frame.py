import cv2
import os

def save_frame_at_timestamp(video_path, timestamp, output_dir, name):
    """
    Extract and save a frame from a video at the specified timestamp.
    
    Args:
        video_path (str): Path to the video file.
        timestamp (str): Timestamp in the format "minutes:seconds" (e.g., "1:23").
        output_dir (str): Directory where the frame image will be saved.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Parse the timestamp
    try:
        minutes, seconds = map(int, timestamp.split(':'))
        target_time_in_seconds = minutes * 60 + seconds
    except ValueError:
        print("Error: Invalid timestamp format. Use 'minutes:seconds' (e.g., '1:23').")
        cap.release()
        return

    # Get the video's frame rate (fps) and total duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_in_seconds = total_frames / fps

    if target_time_in_seconds > duration_in_seconds:
        print(f"Error: Timestamp exceeds video duration ({duration_in_seconds:.2f} seconds).")
        cap.release()
        return

    # Calculate the target frame number
    target_frame = int(target_time_in_seconds * fps)

    # Set the video to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the frame at the specified timestamp.")
        cap.release()
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the frame to disk
    output_file = os.path.join(output_dir, f"{name}_{minutes}m{seconds}s.jpg")
    cv2.imwrite(output_file, frame)

    print(f"Frame saved at {output_file}")

    # Release the video capture object
    cap.release()

# Example usage
# video_path = '/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-green_cabinat-videos/demo-green_cabinet-formal4.mov'  # Replace with the path to your video file
# name = 'green_cabinet'
# timestamp = '1:57'  # Replace with your desired timestamp
# output_dir = './data/real_world/video_frames'  # Replace with your desired output directory

video_path = '/media/yufei/42b0d2d4-94e0-45f4-9930-4d8222ae63e51/yufei/projects/chialiang-real-world-results/formal-high-8_low-96-grey_drawer-videos/formal-demo-grey_drawer-formal1.MOV'  # Replace with the path to your video file
name = 'grey_drawer'
timestamp = '0:0'  # Replace with your desired timestamp
# timestamp = '0:36'  # Replace with your desired timestamp
# timestamp = '2:55'  # Replace with your desired timestamp
output_dir = './data/real_world/video_frames'  # Replace with your desired output directory

save_frame_at_timestamp(video_path, timestamp, output_dir, name)
