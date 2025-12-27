import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed


SKIP_NAMES = {
    "action_dist",
    "demo_rgbs",
    "all_demo_path.txt",
    "meta_info.json",
    "example_pointcloud",
}


def _list_indexed_pkls(traj_path: str):
    """Return sorted integer indices for files like 0.pkl, 1.pkl, ..."""
    idxs = []
    for fn in os.listdir(traj_path):
        if not fn.endswith(".pkl"):
            continue
        stem = fn[:-4]
        if stem.isdigit():
            idxs.append(int(stem))
    idxs.sort()
    return idxs


def remove_duplicate_reset_traj_folder(traj_path: str, target_path: str):
    """
    Finds the last segment where goal_gripper_pcd is identical to the final goal,
    and copies that segment to a sibling folder with 'reset' -> 'reset_only'.
    """
    new_path = target_path + "/".join(traj_path.split("/")[-2:])
    os.makedirs(new_path, exist_ok=True)

    idxs = _list_indexed_pkls(traj_path)
    if len(idxs) == 0:
        return {"traj_path": traj_path, "status": "skip_empty"}

    # Assumes trajectory is contiguous 0..N-1, but we still handle if not perfectly contiguous.
    last_idx = idxs[-1]

    with open(os.path.join(traj_path, f"{last_idx}.pkl"), "rb") as f:
        data = pkl.load(f)
        last_goal_gripper = data["goal_gripper_pcd"]

    last_t_time = None
    # Walk backwards from second-to-last available index
    for t in reversed(idxs[:-1]):
        with open(os.path.join(traj_path, f"{t}.pkl"), "rb") as f:
            data = pkl.load(f)

        diff = np.abs(data["goal_gripper_pcd"] - last_goal_gripper).sum()
        if diff > 1e-3:
            last_t_time = t
            break

    # If never found a "different" one, treat beg as the first index
    if last_t_time is None:
        beg_t = idxs[0]
    else:
        beg_t = last_t_time

    # Copy files 0 ... beg_t (only those that exist in idxs)
    # print(f"Copying traj {traj_path} to {new_path}, keeping 0 to {beg_t} (out of {last_idx})")
    copy_idxs = [t for t in idxs if t <= beg_t]
    for out_i, t in enumerate(copy_idxs):
        src = os.path.join(traj_path, f"{t}.pkl")
        dst = os.path.join(new_path, f"{out_i}.pkl")
        shutil.copy2(src, dst)

    return {
        "traj_path": traj_path,
        "status": "ok",
        "beg_t": beg_t,
        "n_in": len(idxs),
        "n_out": len(copy_idxs),
        "out_path": new_path,
    }


def _collect_traj_paths(root_path: str):
    obj_folders = sorted(os.listdir(root_path))
    traj_paths = []
    for obj_folder in obj_folders:
        obj_path = os.path.join(root_path, obj_folder)
        if not os.path.isdir(obj_path):
            continue

        traj_folders = sorted(os.listdir(obj_path))
        traj_folders = [x for x in traj_folders if x not in SKIP_NAMES]

        for traj_folder in traj_folders:
            traj_path = os.path.join(obj_path, traj_folder)
            if os.path.isdir(traj_path) and 'grasp_only' not in traj_path:
                traj_paths.append(traj_path)

    return traj_paths


def remove_duplicate_reset_whole_folder(root_path: str, target_path: str, num_workers: int = None):
    traj_paths = _collect_traj_paths(root_path)
    if len(traj_paths) == 0:
        print(f"No trajectory folders found under {root_path}")
        return

    print("Found {} trajectory folders to process.".format(len(traj_paths)))
    # traj_paths = traj_paths[:5]

    # Default: use CPU count (or let executor decide if None)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(remove_duplicate_reset_traj_folder, p, target_path) for p in traj_paths]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing traj folders"):
            try:
                res = fut.result()
                # If you want occasional logs:
                # if res.get("status") != "ok":
                #     print(res)
            except Exception as e:
                # Don’t crash the whole run; surface which folder failed
                print(f"[ERROR] One trajectory failed: {e}")


if __name__ == "__main__":
    # root_path = "/tmp/165-obj_reset_1203/"
    root_path = "/tmp/articubot_all_reset_1203/"
    root_path = "/tmp/invert_push_reset/"
    
    root_path = '/tmp/pick_and_place/top_cgn_grasp/'
    target_path = "/tmp/pick_and_place/top_cgn_grasp_grasp_only/"
    
    root_path = '/tmp/pick_and_place/inside_whole_cgn_grasp/'
    target_path = "/tmp/pick_and_place/inside_whole_cgn_grasp_grasp_only/"
    
    root_path = '/tmp/pick_and_place/inside_link_cgn_grasp/'
    target_path = "/tmp/pick_and_place/inside_link_cgn_grasp_grasp_only/"
    remove_duplicate_reset_whole_folder(root_path, target_path, num_workers=6)
