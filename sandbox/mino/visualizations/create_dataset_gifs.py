#!/usr/bin/env python3
import h5py
import typer
from pathlib import Path
import imageio
from third_party.robogen.robogen_utils import compute_new_goal_gripper_pcd
import numpy as np
def create_gifs(
    h5_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="Path to the HDF5 file"),
    output_dir: Path = typer.Option(None, help="Directory to save GIFs (defaults to HDF5 file's parent)"),
    fps: int = typer.Option(10, help="Frames per second for the GIFs"),
):
    """
    Read agentview_image sequences from each demo in the HDF5 file and save as GIFs.
    """
    # Determine where to save GIFs
    if output_dir is None:
        output_dir = h5_path.parent / 'gifs'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open HDF5 file
    with h5py.File(h5_path, 'r') as f:
        data_group = f.get('data') or f
        # Iterate over demo groups
        for demo_name in sorted(data_group.keys()):
            demo_grp = data_group[demo_name]
            if demo_name not in ('demo_656', 'demo_652', 'demo_780', 'demo_601', 'demo_397', 'demo_961', 'demo_913', 'demo_577', 'demo_530', 'demo_119'):
                continue
            if 'obs' not in demo_grp:
                typer.echo(f"Skipping {demo_name}: no 'obs' group found.")
                continue
            obs_grp = demo_grp['obs']
            if 'agentview_image' not in obs_grp:
                typer.echo(f"Skipping {demo_name}: no 'agentview_image' dataset found.")
                continue

            imgs = []
            compute_new_goal_gripper_pcd(np.zeros((200,4,3)), obs_grp['robot0_gripper_qpos'][()], demo_grp['actions'])
            # agentview_image expected shape: (T, H, W, C) or (T, C, H, W)
            arr = obs_grp['agentview_image']
            for i in range(arr.shape[0]):
                frame = arr[i]
                # Move channel axis if needed
                if frame.ndim == 3 and frame.shape[0] in (1, 3):
                    # assume (C, H, W)
                    frame = frame.transpose(1, 2, 0)
                imgs.append(frame)

            # Save GIF
            gif_path = output_dir / f"{h5_path.stem}_{demo_name}.gif"
            imageio.mimsave(str(gif_path), imgs, fps=fps)
            typer.echo(f"Saved GIF for {demo_name} -> {gif_path}")


def main():
    typer.run(create_gifs)

if __name__ == '__main__':
    main()
