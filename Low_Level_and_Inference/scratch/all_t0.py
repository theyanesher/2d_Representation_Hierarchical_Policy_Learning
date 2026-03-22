import argparse
from pathlib import Path
import imageio.v3 as iio
from PIL import Image
import numpy as np

def create_video_from_gifs(root_folder, output_name, limit, fps):
    root = Path(root_folder)
    
    if not root.exists():
        print(f"Error: The directory '{root}' does not exist.")
        return

    # Find all 'all.gif' files recursively
    print(f"Scanning {root}...")
    gif_paths = list(root.rglob("all.gif"))
    
    if not gif_paths:
        print("No 'all.gif' files found.")
        return

    # Apply the user-defined limit
    if limit:
        gif_paths = gif_paths[:limit]

    frames = []
    target_size = None

    for gif_path in gif_paths:
        try:
            with Image.open(gif_path) as img:
                img.seek(0)  # Grab first frame
                frame_rgb = img.convert("RGB")
                
                # Standardize size based on the first frame found
                if target_size is None:
                    target_size = frame_rgb.size
                    print(f"Video resolution set to: {target_size}")
                
                if frame_rgb.size != target_size:
                    frame_rgb = frame_rgb.resize(target_size, Image.Resampling.LANCZOS)
                
                frames.append(np.array(frame_rgb))
                print(f"Processed: {gif_path.relative_to(root)}")
                
        except Exception as e:
            print(f"Skipping {gif_path} due to error: {e}")

    if frames:
        output_file = Path(output_name)
        print(f"Encoding {len(frames)} frames into {output_file}...")
        iio.imwrite(output_file, frames, fps=fps, codec="libx264")
        print("Successfully created video!")
    else:
        print("No valid frames were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile first frames of 'all.gif' files into a video.")
    
    # Positional argument (Required)
    parser.add_argument("directory", type=str, help="The root directory to search")
    
    # Optional arguments
    parser.add_argument("--output", "-o", type=str, default="compilation.mp4", help="Output filename (default: compilation.mp4)")
    parser.add_argument("--limit", "-n", type=int, default=100, help="Stop after n subdirectories")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video")

    args = parser.parse_args()

    create_video_from_gifs(args.directory, args.output, args.limit, args.fps)