import os
import imageio
import zarr
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


data_list = [
    # "outputs_1st_close_jar",
    # "outputs_1st_place_shape_in_shape_sorter",
    # "outputs_1st_reach_and_drag",
    # "outputs_1st_place_wine_at_rack_location",
    # "outputs_1st_slide_block_to_color_target",
    # "outputs_1st_light_bulb_in",                             ######## VERY VERY IMP IMP (Recheck what happened)
    # "outputs_1st_push_buttons",                              ######## VERY VERY IMP IMP  (Recheck what happened)
    "outputs_1st_meat_off_grill",
    "outputs_1st_put_groceries_in_cupboard",
    "outputs_1st_stack_blocks",
    "outputs_1st_open_drawer",
    "outputs_1st_put_item_in_drawer",
    "outputs_1st_sweep_to_dustpan_of_size",
    "outputs_1st_place_cups",
    "outputs_1st_put_money_in_safe",
]

for data in data_list:


    OUT = f"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/{data}/trajectories_data_FINAL/"
    ROOT = f"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/RVT2_Codebase/RVT_Backup_IDK/RVT/data_diffusion_policy/{data}/"
    episodes = sorted(os.listdir(os.path.join(ROOT, "heatmap_images_dist_from_rvt")))
    for ep in tqdm(episodes):  # same episodes list you already use
        ep_zarr_path = os.path.join(OUT, f"{ep}.zarr")
        # import pdb; pdb.set_trace();
        # Open EXISTING zarr (append mode)
        ep_out = zarr.open(ep_zarr_path, mode="a")

        # Skip if already created (safe guard)
        if "heatmap_dist_rvt" in ep_out:
            print(f"Skipping {ep}, heatmap_dist_rvt already exists")
            continue

        hm_dist_grp = ep_out.create_group("heatmap_dist_rvt")
        # import pdb; pdb.set_trace();
        # Infer T exactly the same way as before
        cam1_dir = os.path.join(
            ROOT, "heatmap_images_dist_from_rvt", ep, "camera1"
        )
        # timesteps = sorted(os.listdir(cam1_dir))
        # import pdb; pdb.set_trace();
        timesteps = sorted(
        os.listdir(cam1_dir),
        key=lambda x: int(os.path.splitext(x)[0])
        )
        # for_timestep_cam1_dir = os.path.join(
        #     ROOT, "unnormalized_rgb", ep, "camera1"
        # )
        # timesteps = sorted(
        # os.listdir(for_timestep_cam1_dir),
        # key=lambda x: int(os.path.splitext(x)[0])
        # )
        T = len(timesteps)
        # import pdb; pdb.set_trace();
        for cam in ["camera1", "camera2", "camera3"]:
            hms = []
            for t in range(T):
                hm = imageio.imread(
                    os.path.join(
                        ROOT,
                        "heatmap_images_dist_from_rvt",
                        ep,
                        cam,
                        f"{t}.png",
                    )
                )
                hms.append(hm)
            # import pdb; pdb.set_trace();
            arr = np.stack(hms, axis=0)  # (T, H, W, 3) or (T, H, W)

            # ---- match your existing pipeline ----
            arr = arr.astype(np.float32) / 255.0

            if arr.ndim == 4:  # RGB → grayscale
                arr = arr.mean(axis=3, keepdims=True)  # (T, H, W, 1)
            else:
                arr = arr[..., None]                   # (T, H, W, 1)

            arr = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            # import pdb; pdb.set_trace();
            arr = F.interpolate(
                arr,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            )
            # import pdb; pdb.set_trace();
            arr = arr.numpy()
            # import pdb; pdb.set_trace();
            hm_dist_grp.create_dataset(
                cam,
                data=arr,
                chunks=(8, *arr.shape[1:]),
                compressor=zarr.Blosc(cname="lz4", clevel=3),
            )
            # import pdb; pdb.set_trace();
