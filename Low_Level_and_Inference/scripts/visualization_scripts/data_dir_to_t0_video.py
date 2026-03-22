import h5py
import numpy as np
import imageio
import typer
from tqdm import tqdm
from pathlib import Path
from typing import Optional

app = typer.Typer()

@app.command()
def generate_t0_video(
    input_dir: str = "data/rgb/41510/",
    output_path: str = "media/all_t0.mp4",
    fps: int = 10,
    max_videos: Optional[int] = None
):
    """
    Generates a single video showing the first timestep (t=0) of every trajectory.
    Each frame is a horizontal concatenation of 3 camera views.
    """
    input_path = Path(input_dir)
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(list(input_path.glob("*.h5")))
    if max_videos is not None:
        h5_files = h5_files[:max_videos]

    if not h5_files:
        typer.echo(f"No .h5 files found in {input_dir}")
        return

    typer.echo(f"Input:  {input_path}")
    typer.echo(f"Output: {save_path}")
    typer.echo(f"Processing {len(h5_files)} trajectories...")

    cam_keys = ['obs/cam0_image', 'obs/cam1_image', 'obs/cam2_image']

    with imageio.get_writer(save_path, fps=fps, macro_block_size=None) as writer:
        for h5_file_path in tqdm(h5_files, desc="Collecting t=0 frames"):
            with h5py.File(h5_file_path, 'r') as f:
                available_keys = [k for k in cam_keys if k in f]
                if len(available_keys) < 3:
                    typer.echo(f"Skipping {h5_file_path.name}: Found {len(available_keys)}/3 cameras.")
                    continue

                combined_frame = np.hstack([f[k][0] for k in cam_keys])
                writer.append_data(combined_frame)

    typer.echo(f"\nDone! Video saved to: {save_path}")

if __name__ == "__main__":
    app()
