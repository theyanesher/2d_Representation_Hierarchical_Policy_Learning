import h5py
import numpy as np
import imageio
import typer
from tqdm import tqdm
from pathlib import Path

app = typer.Typer()

@app.command()
def generate_batch_videos(
    input_dir: str = "data/rgb_camera_randomized/41510/",
    max_videos: int = 100,
    fps: int = 20
):
    """
    Automatically maps data/ to media/ and generates horizontal 3-camera RGB videos.
    """
    input_path = Path(input_dir)
    
    # Logic to swap 'data' for 'media' in the path
    # This works even if the sub-folders change
    parts = list(input_path.parts)
    if parts[0] == 'data':
        parts[0] = 'media'
    else:
        # Fallback if 'data' isn't the root
        parts.insert(0, 'media')
    
    output_dir = Path(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .h5 files
    h5_files = sorted(list(input_path.glob("*.h5")))
    files_to_process = h5_files[:max_videos]
    
    if not files_to_process:
        typer.echo(f"No .h5 files found in {input_dir}")
        return

    typer.echo(f"Input:  {input_path}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Processing {len(files_to_process)} videos...")

    for h5_file_path in files_to_process:
        video_name = h5_file_path.stem + ".mp4"
        save_path = output_dir / video_name

        with h5py.File(h5_file_path, 'r') as f:
            cam_keys = ['obs/cam0_image', 'obs/cam1_image', 'obs/cam2_image']
            
            # Verify keys exist
            available_keys = [k for k in cam_keys if k in f]
            if len(available_keys) < 3:
                typer.echo(f"Skipping {h5_file_path.name}: Found {len(available_keys)}/3 cameras.")
                continue

            num_frames = f[cam_keys[0]].shape[0]

            with imageio.get_writer(save_path, fps=fps, macro_block_size=None) as writer:
                for t in tqdm(range(num_frames), desc=f"🎥 {video_name}", leave=False):
                    # Concatenate cam0, cam1, and cam2 side-by-side
                    combined_frame = np.hstack([f[k][t] for k in cam_keys])
                    writer.append_data(combined_frame)

    typer.echo(f"\nDone! Videos are located in: {output_dir}")

if __name__ == "__main__":
    app()