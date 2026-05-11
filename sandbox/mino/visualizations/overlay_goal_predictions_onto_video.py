import imageio
import cv2
import pickle
import numpy as np
from pathlib import Path
import argparse

def project_points(points: np.ndarray,
                   intrinsic: np.ndarray,
                   extrinsic: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    cam_pts = (extrinsic @ pts_h.T)[:3, :]
    pix = intrinsic @ cam_pts
    pix[:2, :] /= pix[2:3, :]
    return pix[:2, :].T

def overlay_points_to_gif(video_path: Path,
                          pickle_dir: Path,
                          intrinsic: np.ndarray,
                          extrinsic: np.ndarray,
                          output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    duration = 1.0 / fps
    writer = imageio.get_writer(str(output_path), mode='I', duration=duration)

    num_pkl_files = len(list(pickle_dir.glob("*.pkl")))
    pkl_files = [pickle_dir / f'{i}.pkl' for i in range(num_pkl_files)]
    frame_idx = 0
    frames_per_pkl = 4
    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_idx >= len(pkl_files) * frames_per_pkl:
            break

        with open(str(pkl_files[frame_idx // frames_per_pkl]), "rb") as f:
            data = pickle.load(f)
            pts = data["subgoal_pred"][-1]

        pix = project_points(pts, intrinsic, extrinsic)
        for u, v in pix:
            cv2.circle(frame_bgr,
                       (int(round(u)), int(round(v))),
                       radius=5,
                       color=(255, 0, 0),
                       thickness=-1)

        # Convert BGR to RGB before writing
        frame_rgb = frame_bgr[..., ::-1]
        writer.append_data(frame_rgb)
        frame_idx += 1

    cap.release()
    writer.close()
    print(f"Finished. Saved GIF to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media_dir", type=Path, required=True,
                        help="Directory containing .mp4 and matching pickle subfolders")
    args = parser.parse_args()
    media_dir: Path = args.media_dir

    # fill in your camera matrices
    agentview_intrinsics = np.loadtxt('sandbox/mino/camera_calibrations/agentview_camera_intrinsics.txt')
    world_to_agentview_extrinsics = np.loadtxt('sandbox/mino/camera_calibrations/agentview_world_to_cam_extrinsics.txt')

    output_dir = media_dir / "gifs"
    output_dir.mkdir(exist_ok=True)

    for video_path in sorted(media_dir.glob("*.mp4")):
        base = video_path.stem
        pickle_subdir = media_dir / "goal_predictions" / base
        gif_path = output_dir / f"{base}_overlay.gif"

        if not pickle_subdir.is_dir():
            print(f"Skipping {video_path.name}: no {pickle_subdir.name}/ directory")
            continue

        overlay_points_to_gif(video_path,
            pickle_subdir,
            agentview_intrinsics,
            world_to_agentview_extrinsics,
            gif_path)

if __name__ == "__main__":
    main()
