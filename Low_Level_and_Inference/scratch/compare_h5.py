import h5py
import matplotlib.pyplot as plt
import numpy as np
import imageio
import typer
from pathlib import Path
from tqdm import tqdm

app = typer.Typer()

def get_images(h5_file, cam_idx):
    """Normalizes image loading from different H5 structures."""
    if 'obs/rgb' in h5_file:
        # Structure A: TxCxHxWx3
        return h5_file['obs/rgb'][:, cam_idx]
    else:
        # Structure B: obs/camX_image
        key = f'obs/cam{cam_idx}_image'
        if key in h5_file:
            return h5_file[key][:]
    return None

@app.command()
def compare_h5_full(
    file_a: str,
    file_b: str,
    output_prefix: str = "media/comparison",
    fps: int = 20
):
    f1 = h5py.File(file_a, 'r')
    f2 = h5py.File(file_b, 'r')

    try:
        # --- PART 1: Plotting Actions & States (20 Plots) ---
        typer.echo("Generating Action/State comparison plots...")
        act1, act2 = f1['action'][:], f2['action'][:]
        state1, state2 = f1['obs/state'][:], f2['obs/state'][:]

        fig, axes = plt.subplots(10, 2, figsize=(15, 30))
        for i in range(10):
            # Actions
            axes[i, 0].plot(act1[:, i], label='File A', alpha=0.7)
            axes[i, 0].plot(act2[:, i], label='File B', ls='--', alpha=0.7)
            axes[i, 0].set_title(f"Action Dim {i}")
            
            # States
            axes[i, 1].plot(state1[:, i], label='File A', color='green', alpha=0.7)
            axes[i, 1].plot(state2[:, i], label='File B', color='orange', ls='--', alpha=0.7)
            axes[i, 1].set_title(f"State Dim {i}")
            
            axes[i, 0].legend(); axes[i, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_data.png")

        # --- PART 2: Generating Side-by-Side Video ---
        typer.echo("Generating side-by-side video comparison...")
        # Get Cam 0 from both (most common reference)
        cam_idx = 1
        imgs1 = get_images(f1, cam_idx)
        imgs2 = get_images(f2, cam_idx)

        if imgs1 is not None and imgs2 is not None:
            num_frames = min(len(imgs1), len(imgs2))
            writer = imageio.get_writer(f"{output_prefix}_cam{cam_idx}_video.mp4", fps=fps)
            
            for t in tqdm(range(num_frames)):
                # Concatenate File A (Left) and File B (Right)
                combined = np.hstack([imgs1[t], imgs2[t]])
                writer.append_data(combined)
            writer.close()
        else:
            typer.echo("Error: Could not find matching camera streams.")

    finally:
        f1.close()
        f2.close()
        typer.echo(f"Done! Saved {output_prefix}_data.png and {output_prefix}_cam{cam_idx}_video.mp4")

if __name__ == "__main__":
    app()