import os
import shutil

SRC_ROOT = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Bimanual_Manipulation/rvt/data/train/insert_onto_square_peg/all_variations/episodes/"
DST_ROOT = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/outputs_1st_insert_onto_square_peg"

def copy_all_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

for episode in sorted(os.listdir(SRC_ROOT)):
    src_episode = os.path.join(SRC_ROOT, episode)

    if not episode.startswith("episode") or not os.path.isdir(src_episode):
        continue

    wrist_depth = os.path.join(src_episode, "wrist_depth")
    wrist_rgb = os.path.join(src_episode, "wrist_rgb")

    if not (os.path.isdir(wrist_depth) and os.path.isdir(wrist_rgb)):
        print(f"Skipping {episode}: missing wrist data")
        continue

    dst_depth_cam4 = os.path.join(DST_ROOT, "depth", episode, "camera4")
    dst_rgb_cam4 = os.path.join(DST_ROOT, "unnormalized_rgb", episode, "camera4")

    print(f"Copying {episode}")
    copy_all_files(wrist_depth, dst_depth_cam4)
    copy_all_files(wrist_rgb, dst_rgb_cam4)
