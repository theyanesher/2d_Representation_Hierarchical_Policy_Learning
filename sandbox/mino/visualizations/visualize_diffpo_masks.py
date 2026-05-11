import imageio
import cv2
import pickle
import numpy as np
from pathlib import Path
import argparse

def overlay_points_to_gif(pickle_dir: Path,
                          output_path: Path,
                          fps = 20):
    writer = imageio.get_writer(str(output_path), mode='I', duration=1.0 / fps)

    num_pkl_files = len(list(pickle_dir.glob("*.pkl")))
    pkl_files = [pickle_dir / f'{i}.pkl' for i in range(num_pkl_files)]
    frame_idx = 0
    frames_per_pkl = 4
    for pkl_file in pkl_files:
        with open(str(pkl_file), "rb") as f:
            data = pickle.load(f)
            frame_rgb = data['agentview_image']
            if frame_rgb.shape[-1] == 4:
                frame_rgb = frame_rgb[...,:3] * 255.0
                depth = frame_rgb[..., -1] * 255.0
                depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
                combined = np.concatenate([frame_rgb, depth_rgb], axis=1).astype(np.uint8)
                writer.append_data(combined)

            elif 'agentview_cond' in data:
                frame_rgb = frame_rgb[...,:3] * 255.0
                flow = data['agentview_cond']
                flow_mask = np.abs(flow).sum(axis=-1) > 0
                ys, xs = np.where(flow_mask)
                for y, x in zip(ys, xs):
                    dx, dy, dz = flow[y, x]
                    start = (int(x), int(y))
                    end   = (int(x + dx), int(y + dy))
                    cv2.arrowedLine(
                        frame_rgb, start, end,
                        color=(0, 255, 0),      # green arrows
                        thickness=1,
                        tipLength=0.3
                    )
                frame_rgb = frame_rgb.astype(np.uint8)
                writer.append_data(frame_rgb)
            else:
                raise ValueError(f'unexpected image shape: {frame_rgb.shape}')

            frame_idx += 1

    writer.close()
    print(f"Finished. Saved GIF to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media_dir", type=Path, required=True,
                        help="Directory containing .mp4 and matching pickle subfolders")
    args = parser.parse_args()
    media_dir: Path = args.media_dir

    output_dir = media_dir / "gifs"
    output_dir.mkdir(exist_ok=True)

    for video_path in sorted(media_dir.glob("*.mp4")):
        base = video_path.stem
        pickle_subdir = media_dir / "goal_predictions" / base
        gif_path = output_dir / f"{base}_overlay.gif"

        if not pickle_subdir.is_dir():
            print(f"Skipping {video_path.name}: no {pickle_subdir.name}/ directory")
            continue

        overlay_points_to_gif(
            pickle_subdir,
            gif_path)

if __name__ == "__main__":
    main()
