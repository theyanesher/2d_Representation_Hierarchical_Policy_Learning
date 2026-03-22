import h5py
import numpy as np
import imageio
import typer
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import cv2

app = typer.Typer()

@app.command()
def generate_video(
    h5_path: str,
    output_path: Optional[str] = None,
    fps: int = 20
):
    """
    Reads an h5 file and creates a horizontally concatenated video of
    the camera observations.
    
    If output_path is not provided, it defaults to the h5 filename with .mp4 extension.
    """
    # specific logic to handle the filename
    if output_path is None:
        # e.g., "data/demonstration_01.h5" -> "demonstration_01.mp4"
        output_path = "media/" + Path(h5_path).stem + ".mp4"

    with h5py.File(h5_path, 'r') as h5_file:
        rgb_cam_keys = []
        depth_cam_keys = []
        for i in range(3):  # Check for cam0, cam1, cam2
            rgb_key = f'obs/cam{i}_image'
            depth_key = f'obs/cam{i}_depth'
            if rgb_key in h5_file:
                rgb_cam_keys.append(rgb_key)
            if depth_key in h5_file:
                depth_cam_keys.append(depth_key)

        if not rgb_cam_keys:
            typer.echo("Error: No RGB camera data found with keys like 'obs/camX_image'.")
            return
            
        if not depth_cam_keys:
            typer.echo("Error: No depth camera data found with keys like 'obs/camX_depth'.")
            return

        num_cams = min(len(rgb_cam_keys), len(depth_cam_keys))
        if len(rgb_cam_keys) != len(depth_cam_keys):
            typer.echo(f"Warning: Mismatch in number of RGB ({len(rgb_cam_keys)}) and depth ({len(depth_cam_keys)}) cameras. Using {num_cams} pairs.")

        rgb_cam_keys = rgb_cam_keys[:num_cams]
        depth_cam_keys = depth_cam_keys[:num_cams]

        num_frames = h5_file[rgb_cam_keys[0]].shape[0]
        
        H, W, _ = h5_file[rgb_cam_keys[0]][0].shape
        black_image = np.zeros((H, W, 3), dtype=np.uint8)

        typer.echo(f"Found {num_cams} camera pairs (RGB and Depth).")
        typer.echo(f"Processing {num_frames} frames from {h5_path}...")
        typer.echo(f"Saving to {output_path}...")

        with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for t in tqdm(range(num_frames)):
                rgb_frames = [h5_file[key][t] for key in rgb_cam_keys]
                depth_frames_raw = [h5_file[key][t] for key in depth_cam_keys]

                while len(rgb_frames) < 3:
                    rgb_frames.append(black_image)
                processed_depth_frames = []
                for depth_frame in depth_frames_raw:
                    d_min, d_max = depth_frame.min(), depth_frame.max()
                    if d_max - d_min > 1e-6:
                        depth_normalized = (depth_frame - d_min) / (d_max - d_min) * 255.0
                    else:
                        depth_normalized = np.zeros_like(depth_frame)
                    
                    depth_display = depth_normalized.astype(np.uint8)
                    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    processed_depth_frames.append(depth_colormap)
                
                while len(processed_depth_frames) < 3:
                    processed_depth_frames.append(black_image)

                top_row = np.hstack(rgb_frames)
                bottom_row = np.hstack(processed_depth_frames)
                combined_frame = np.vstack([top_row, bottom_row])
                
                writer.append_data(combined_frame)
            
    typer.echo(f"Done.")

if __name__ == "__main__":
    app()