import os
import subprocess
from multiprocessing import Pool, cpu_count
import zipfile

# target_path = "/project/flame/yufeiw2/RoboGen-sim2real/data/dp3_demo"
# target_path = "/project/flame/yufeiw2/RoboGen-sim2real/data/acronym/renders"
target_path = "/tmp/165-obj_reset_only_1203"
target_path = "/tmp/articubot_all_reset_only_1203"
target_path = "/tmp/invert_push_reset_only"

target_path = "/tmp/pick_and_place/inside_link_cgn_grasp_grasp_only"
# target_path = "/tmp/pick_and_place/inside_whole_cgn_grasp_grasp_only"
# target_path = "/tmp/pick_and_place/top_cgn_grasp_grasp_only"

target_path = "/tmp/pick_and_place/inside_link_cgn_1204_place_only"
# target_path = "/tmp/pick_and_place/inside_whole_cgn_1204_place_only"
# target_path = "/tmp/pick_and_place/top_cgn_1204_place_only"
all_obj_trajs = sorted(os.listdir(target_path))
max_parallel_jobs = 12  # Adjust this to limit parallel zipping

def zip_folder(obj_traj):
    folder_path = os.path.join(target_path, obj_traj)
    zip_path = os.path.join(target_path, f"{obj_traj}.zip")
    if os.path.exists(zip_path):
        return obj_traj, -1, "already zipped"

    if not os.path.isdir(folder_path):
        return obj_traj, -1, "Not a directory"

    try:
        print("trying to zip {}".format(folder_path))
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    abs_path = os.path.join(root, file)
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