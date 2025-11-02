import os
import subprocess
from multiprocessing import Pool, cpu_count
import zipfile

# target_path = "/project_data/held/chenyuah/RoboGen-sim2real/data/dp3_demo/random_cam"
# saving_zip_path = "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/new_7_category_random_cam"

# target_path = "/project_data/held/chenyuah/RoboGen-sim2real/data/dp3_demo/real_world_cam"
# saving_zip_path =  "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/new_7_category_real_cam"

target_path = "/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal"
saving_zip_path =  "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/dp3_demo_real_world_noise_pcd_clean_distorted_goal"

target_path = "/scratch/yufeiw2/dp3_demo_clean_distorted_goal"
saving_zip_path =  "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/dp3_demo_clean_distorted_goal"

target_path = "/scratch/chenyuah/invert_push"
saving_zip_path =  "/project_data/held/yufeiw2/articubot_multitask/RoboGen-sim2real/data/invert_push"

if not os.path.exists(saving_zip_path):
    os.makedirs(saving_zip_path)

all_obj_trajs = sorted(os.listdir(target_path))
max_parallel_jobs = 8  # Adjust this to limit parallel zipping

def zip_folder(obj_traj):
    folder_path = os.path.join(target_path, obj_traj)
    zip_path = os.path.join(saving_zip_path, f"{obj_traj}.zip")  # save into saving_zip_path
    if os.path.exists(zip_path):
        return obj_traj, -1, "already zipped"

    if not os.path.isdir(folder_path):
        return obj_traj, -1, "Not a directory"

    try:
        print(f"trying to zip {folder_path} -> {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    abs_path = os.path.join(root, file)
                    # keep paths relative to target_path so folder structure is preserved
                    rel_path = os.path.relpath(abs_path, target_path)
                    zipf.write(abs_path, rel_path)
        return obj_traj, 0, ""
    except Exception as e:
        return obj_traj, -1, str(e)

if __name__ == "__main__":
    import time
    beg = time.time()
    with Pool(processes=max_parallel_jobs) as pool:
        results = pool.map(zip_folder, all_obj_trajs)
    end = time.time()
    print(f"zipping costs {end - beg}")

    for obj_traj, returncode, err in results:
        if returncode != 0:
            print(f"[ERROR] Failed to zip {obj_traj}: {err.strip()}")
        else:
            print(f"[OK] Zipped {obj_traj}")
