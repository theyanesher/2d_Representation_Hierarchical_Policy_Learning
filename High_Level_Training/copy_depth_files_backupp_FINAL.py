import os
import shutil
import cv2
import numpy as np
import pickle
# from libs.peract_colab.peract_colab.rlbench.utils import get_stored_demo
SRC_ROOT = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Bimanual_Manipulation/rvt/data/train/sweep_to_dustpan_of_size/all_variations/episodes/"
DST_ROOT = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_sweep_to_dustpan_of_size"

MAX_24BIT = 16777215.0
CAMERA_NAME = "camera4"   # name used inside misc dict

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Return a dummy object that accepts any constructor args
        return lambda *args, **kwargs: {}


def copy_rgb_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

def convert_depth_with_near_far(
    depth_file,
    low_dim_obs_file_path,
    dst_path,
    camera
):
    # --- decode depth ---
    depth_image_uint8 = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if depth_image_uint8 is None:
        raise RuntimeError(f"Failed to read depth file: {depth_file}")

    B = depth_image_uint8[:, :, 0].astype(np.uint32)
    G = depth_image_uint8[:, :, 1].astype(np.uint32)
    R = depth_image_uint8[:, :, 2].astype(np.uint32)

    depth_encoded = (R << 16) | (G << 8) | B
    depth_normalized = depth_encoded.astype(np.float32) / MAX_24BIT

    # --- load pickle ---
    # import pdb; pdb.set_trace()
    # demo = get_stored_demo(data_path=data_path, index=d_idx)
    with open(low_dim_obs_file_path, "rb") as f:
        data = pickle.load(f)
    # with open(low_dim_obs_file_path, "rb") as f:
    #     data = SafeUnpickler(f).load()

    # data = safe_pickle_load(low_dim_obs_file_path)

    first_obs = data[0]
    ms = first_obs.misc
    # import pdb; pdb.set_trace()
    far = ms[f"wrist_camera_far"]
    near = ms[f"wrist_camera_near"]
    # import pdb; pdb.set_trace();
    # --- convert to metric depth ---
    depth_image = (depth_normalized * (far - near) + near) * 255
    depth_image = depth_image.astype(np.uint8)

    # --- save ---
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, depth_image)

for episode in sorted(os.listdir(SRC_ROOT)):
    src_episode = os.path.join(SRC_ROOT, episode)
    if not episode.startswith("episode") or not os.path.isdir(src_episode):
        continue

    wrist_depth_dir = os.path.join(src_episode, "wrist_depth")
    wrist_rgb_dir = os.path.join(src_episode, "wrist_rgb")

    if not (os.path.isdir(wrist_depth_dir) and os.path.isdir(wrist_rgb_dir)):
        print(f"Skipping {episode}: missing wrist data")
        continue

    # adjust this if your pickle path differs
    low_dim_obs_file_path = os.path.join(
        src_episode, "low_dim_obs.pkl"
    )

    if not os.path.isfile(low_dim_obs_file_path):
        raise RuntimeError(f"Missing pickle for {episode}")

    dst_depth_dir = os.path.join(DST_ROOT, "depth", episode, "camera4")
    dst_rgb_dir = os.path.join(DST_ROOT, "unnormalized_rgb", episode, "camera4")

    print(f"Processing {episode}")

    # --- depth ---
    for fname in sorted(os.listdir(wrist_depth_dir)):
        src_depth = os.path.join(wrist_depth_dir, fname)
        dst_depth = os.path.join(dst_depth_dir, fname)

        if os.path.isfile(src_depth):
            convert_depth_with_near_far(
                src_depth,
                low_dim_obs_file_path,
                dst_depth,
                CAMERA_NAME
            )

    # --- rgb ---
    copy_rgb_files(wrist_rgb_dir, dst_rgb_dir)
