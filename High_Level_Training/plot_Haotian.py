import os
import cv2
import numpy as np
from matplotlib import cm
from tqdm import tqdm
import re
from PIL import Image

def normalize_to_01(x):
    """Normalize heatmap to [0, 1]."""
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-9:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def apply_jet_colormap(heat):
    """
    heat: H,W or H,W,1
    returns: H,W,3 float32 in [0,1]
    """
    if heat.ndim == 3 and heat.shape[2] == 1:
        heat = heat[:, :, 0]

    heat_norm = normalize_to_01(heat)
    heat_color = cm.jet(heat_norm)[:, :, :3]  # drop alpha, keep RGB
    return heat_color.astype(np.float32)


def blend_rgb_heatmap(rgb, heat_color, alpha=0.6):
    """
    rgb: uint8 H,W,3 (0-255)
    heat_color: float32 H,W,3 (0-1)
    returns float32 H,W,3
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    blended = (1 - alpha) * rgb_f + alpha * heat_color
    return blended


def build_video_writer(save_path, width, height, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(save_path, fourcc, fps, (width, height))


def visualize_folder(
    video_folder, heatmap_folder, save_folder, task, fps=20, alpha=0.6
):
    """
    video_folder: path to RGB frames
    heatmap_folder: path to heatmap .npy files
    save_folder: where the mp4 videos will be saved
    tasks: list of task names
    """
    
    os.makedirs(save_folder, exist_ok=True)

    # loop over tasks
    # for task in tasks:
    # import pdb; pdb.set_trace();
    task_rgb_dir = os.path.join(video_folder)
    task_heat_dir = os.path.join(heatmap_folder)
    os.makedirs(os.path.join(save_folder), exist_ok=True)
    # import pdb; pdb.set_trace();
    # each model is a subfolder: task/{model_name}
    model_names = sorted(os.listdir(task_rgb_dir), key=lambda x: int(re.findall(r"\d+", x)[0]))

    for model_name in model_names:
        # import pdb; pdb.set_trace();
        rgb_path = os.path.join(task_rgb_dir, model_name)
        heat_path = os.path.join(task_heat_dir, model_name)

        if not os.path.isdir(rgb_path) or not os.path.isdir(heat_path):
            continue
        # import pdb; pdb.set_trace();
        # collect frames sorted numerically
        rgb_frames = sorted(
            [f for f in os.listdir(rgb_path) if f.endswith(".png") or f.endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        # import pdb; pdb.set_trace();
        heatmaps = sorted(
            [f for f in os.listdir(heat_path) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        # import pdb; pdb.set_trace();
        if len(rgb_frames) == 0 or len(heatmaps) == 0:
            print(f"Skipping {task}/{model_name} (no frames or heatmaps).")
            continue
        # import pdb; pdb.set_trace();
        # prepare mp4 writer
        sample_rgb = cv2.imread(os.path.join(rgb_path, rgb_frames[0]))
        h, w = sample_rgb.shape[:2]
        # import pdb; pdb.set_trace()
        # save_path = os.path.join(save_folder, task, f"{model_name}.mp4")
        output_task_folder = os.path.join(save_folder, task)
        os.makedirs(output_task_folder, exist_ok=True)
        save_path = os.path.join(output_task_folder, f"{model_name}.mp4")
        writer = build_video_writer(save_path, w, h, fps=fps)
        # import pdb; pdb.set_trace();
        print(f"Processing {task}/{model_name} -> {save_path}")

        for rgb_file, heat_file in tqdm(zip(rgb_frames, heatmaps),
                                        total=min(len(rgb_frames), len(heatmaps))):
            # load RGB
            
            rgb = cv2.imread(os.path.join(rgb_path, rgb_file), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # import pdb; pdb.set_trace();
            # load heatmap
            # heat_np = np.load(os.path.join(heat_path, heat_file))  # H,W or H,W,1

            heat_img = Image.open(os.path.join(heat_path, heat_file)).convert("RGB")
            heat_float = np.array(heat_img).astype(np.float32)

            # Convert RGB -> grayscale by averaging
            heat_gray = heat_float.mean(axis=2)   # shape (H, W)

            # Normalize to [0, 1]
            heat_np = (heat_gray - heat_gray.min()) / (heat_gray.max() - heat_gray.min() + 1e-8)

            # convert heatmap using jet
            heat_color = apply_jet_colormap(heat_np)

            # blend
            blended = blend_rgb_heatmap(rgb, heat_color, alpha=alpha)

            # convert back to BGR for video writer
            blended_bgr_uint8 = (blended * 255).astype(np.uint8)
            blended_bgr_uint8 = cv2.cvtColor(blended_bgr_uint8, cv2.COLOR_RGB2BGR)

            writer.write(blended_bgr_uint8)

        writer.release()
        print(f"Saved video: {save_path}")


# Example call:
# visualize_folder(
#     video_folder="outputs_11th/unnormalized_rgb/episode15/",
#     heatmap_folder="outputs_11th/unnormalized_heatmap_images/episode15",
#     save_folder="output_videos_overlay_Haotian_11_episode15",
#     task="insert_onto_square_peg",
#     fps=20,
#     alpha=0.6,
# )
for ep in range(5):
    visualize_folder(
        video_folder=f"data_diffusion_policy/outputs_1st_close_jar/unnormalized_rgb/episode{ep}/",
        heatmap_folder=f"data_diffusion_policy/outputs_1st_close_jar/unnormalized_heatmap_images/episode{ep}/",
        save_folder=f"plot_data_diffusion_policy_ORIGINAL_HEATMAP/outputs_1st_close_jar/output_videos_overlay_Haotian_2nd_Heatmap{ep}",
        task="close_jar",
        fps=20,
        alpha=0.6,
    )
# rgb_base = "outputs_7th/unnormalized_rgb/episode0"
# heatmap_base = "outputs_7th/unnormalized_heatmap_images/episode0"
# save_dir = "output_videos_overlay"
