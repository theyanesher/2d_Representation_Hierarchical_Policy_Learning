import h5py
import pickle
import typer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import imageio
import cv2
from tqdm import tqdm

app = typer.Typer()
ROOT = Path('/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/')

@app.command('hdf5-images-to-video')
def hdf5_images_to_video(
    task: str = typer.Option(..., help='Task name'),
    demo: int = typer.Option(0, help='Demo index'),
    diffpo: bool = typer.Option(False, help='Use diffusion-policy file'),
    fps: float = typer.Option(10.0, help='Frames per second')
):
    """
    Read one HDF5 demo and save it as a video (MP4) using imageio.
    """
    if diffpo:
        path = ROOT / task / f'{task}_pcd_abs_images_flow.hdf5'
    else:
        path = ROOT / task / f'{task}_abs.hdf5'
    out_dir = ROOT / task / f'{task}_videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = out_dir / f'episode_{demo}.mp4'
    writer = imageio.get_writer(str(vid), fps=fps, codec='libx264')
    with h5py.File(path, 'r') as f:
        grp = f['data'][f'demo_{demo}']['obs']
        length = len(grp['point_cloud'])

        for t in range(length):
            if diffpo:
                a = grp['agentview_image_84'][t][...,:3]
                b = grp['robot0_eye_in_hand_image_84'][t][...,:3]
                a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
                # --- load displacement fmasks (shape H×W×2) ---
                disp_a = grp['agentview_cond_84'][t][...,:2]      # dx, dy
                disp_b = grp['robot0_eye_in_hand_cond_84'][t][...,:2]

                # --- overlay arrows onto 'a' ---
                mask_a = np.any(disp_a != 0, axis=-1)
                ys, xs = np.where(mask_a)
                for y, x in zip(ys, xs):
                    dx, dy = disp_a[y, x]
                    start = (int(x), int(y))
                    end   = (int(x + dx), int(y + dy))
                    cv2.arrowedLine(
                        a, start, end,
                        color=(0, 255, 0),      # green arrows
                        thickness=1,
                        tipLength=0.3
                    )

                # --- overlay arrows onto 'b' ---
                mask_b = np.any(disp_b != 0, axis=-1)
                ys, xs = np.where(mask_b)
                for y, x in zip(ys, xs):
                    dx, dy = disp_b[y, x]
                    start = (int(x), int(y))
                    end   = (int(x + dx), int(y + dy))
                    cv2.arrowedLine(
                        b, start, end,
                        color=(0, 255, 0),
                        thickness=1,
                        tipLength=0.3
                    )


                # 2×2 mosaic
                frame    = np.concatenate((a, b), axis=1)
                # bottom = np.concatenate((disp_a, disp_b), axis=1)
                # frame  = np.concatenate((top, bottom), axis=0)
            else:
                frame = grp['agentview_image'][t]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()

@app.command('hdf5-flow-to-video')
def hdf5_flow_to_video(
    task: str = typer.Option(..., help='Task name'),
    demo: int = typer.Option(0, help='Demo index'),
    fps: float = typer.Option(10.0, help='Frames per second')
):
    path    = ROOT / task / f'{task}_pcd_abs_images_dddxyz_camera_1.hdf5'
    out_dir = ROOT / task / 'flow_videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_fp  = str(out_dir / f'episode_{demo}.mp4')

    # First pass: figure out frame size
    with h5py.File(path, 'r') as f:
        grp    = f['data'][f'demo_{demo}']['obs']
        H, W   = grp['agentview_image_84'].shape[1:3]
        grid_w = 4 * W
        grid_h = 2 * H

    # OpenCV writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'/'H264'
    writer = cv2.VideoWriter(vid_fp, fourcc, fps, (grid_w, grid_h))

    def to_heatmap(ch):
        # normalize to 0–255 uint8
        ch_u8 = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # apply a nice colormap
        return cv2.applyColorMap(ch_u8, cv2.COLORMAP_VIRIDIS)

    with h5py.File(path, 'r') as f:
        grp   = f['data'][f'demo_{demo}']['obs']
        length = len(grp['point_cloud'])

        for t in tqdm(range(length), desc='Writing video'):
            # load images
            a      = grp['agentview_image_84'][t][...,:3]
            b      = grp['robot0_eye_in_hand_image_84'][t][...,:3]
            disp_a = grp['agentview_cond_84'][t].transpose(1,2,0)
            disp_b = grp['robot0_eye_in_hand_cond_84'][t].transpose(1,2,0)

            # convert to BGR uint8
            a_bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            b_bgr = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)

            # make heatmaps for each channel
            ha = [ to_heatmap(disp_a[..., i]) for i in range(3) ]
            hb = [ to_heatmap(disp_b[..., i]) for i in range(3) ]

            # stitch rows
            top_row    = np.hstack([a_bgr, *ha])
            bottom_row = np.hstack([b_bgr, *hb])
            grid       = np.vstack([top_row, bottom_row])

            writer.write(grid)

    writer.release()
    print(f"Saved video to {vid_fp}")

if __name__ == '__main__':
    app()
