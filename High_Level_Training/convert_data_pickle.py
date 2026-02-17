import os
import re
import zarr
import numpy as np
import imageio
import pickle
from tqdm import tqdm
import torch
import clip
import torch.nn.functional as F
import numpy as np
from all_types_of_verifications.clip_model_verification import _clip_encode_text

data_list = [
    # "outputs_1st_close_jar",
    # "outputs_1st_place_shape_in_shape_sorter",
    # "outputs_1st_reach_and_drag",
    # "outputs_1st_place_wine_at_rack_location",
    # "outputs_1st_slide_block_to_color_target",
    # "outputs_1st_light_bulb_in",
    # "outputs_1st_push_buttons",
    # "outputs_1st_meat_off_grill",
    # "outputs_1st_put_groceries_in_cupboard",
    # "outputs_1st_stack_blocks",
    # "outputs_1st_open_drawer",
    # "outputs_1st_put_item_in_drawer",
    "outputs_1st_sweep_to_dustpan_of_size",
    "outputs_1st_place_cups",
    "outputs_1st_put_money_in_safe",
]

for data in data_list:

    ROOT = f"data_diffusion_policy/{data}"    #close_jar" #"/path/to/data"
    OUT = f"data_diffusion_policy/{data}/trajectories_data_FINAL" #"/path/to/trajectories"

    original_data = data.replace("outputs_1st_", "")
    # import pdb; pdb.set_trace();
    #folder = "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Bimanual_Manipulation/replay/replay_train/diffusion_policy_dataset/insert_onto_square_peg"
    folder = f"/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/RVT2_GMM_Codebase/Bimanual_Manipulation/rvt/data/train/{original_data}/all_variations/episodes/"
    # replay_folders = [
    #     os.path.join(folder, f)
    #     for f in os.listdir(folder)
    #     if os.path.isdir(os.path.join(folder, f)) and re.match(r'episode\d+$', f)
    # ]


    # import pdb; pdb.set_trace();

    # import pdb; pdb.set_trace();
    # import pdb; pdb.set_trace();
    # for replay_file in replay_files:
    #     with open(replay_file, "rb") as f:
    #         d = pickle.load(f)
    #         print(d['episode_idx'], d['sample_frame'], d["lang_goal"])
    # import pdb; pdb.set_trace();


    episodes = sorted(os.listdir(os.path.join(ROOT, "unnormalized_rgb")))

    for ep in tqdm(episodes):
        ep_out = zarr.open(
            os.path.join(OUT, f"{ep}.zarr"),
            mode="w"
        )

        # Infer timesteps
        cam1_dir = os.path.join(ROOT, "unnormalized_rgb", ep, "camera1")
        timesteps = sorted(os.listdir(cam1_dir))
        T = len(timesteps)

        # ---- RGB ----
        rgb_grp = ep_out.create_group("rgb")
        for cam in ["camera1", "camera2", "camera3", "camera4"]:
            imgs = []
            for t in range(T):
                img = imageio.imread(
                    os.path.join(ROOT, "unnormalized_rgb", ep, cam, f"{t}.png")
                )
                imgs.append(img)
            arr = np.stack(imgs, axis=0)  # (T, H, W, 3)
            # import pdb; pdb.set_trace();
            ################ ADDING TO REDUCE DATALOADER LATENCY ######################
            arr = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            arr = F.interpolate(
                    arr,          # (1, 3, H, W)
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=False
                )
            arr = arr.numpy()
            rgb_grp.create_dataset(
                cam,
                data=arr,
                chunks=(8, *arr.shape[1:]),
                compressor=zarr.Blosc(cname="lz4", clevel=3)
            )

        # ---- Depth ----
        depth_grp = ep_out.create_group("depth")
        for cam in ["camera1", "camera2", "camera3", "camera4"]:
            # import pdb; pdb.set_trace()
            depths = []
            for t in range(T):
                d = imageio.imread(
                    os.path.join(ROOT, "depth", ep, cam, f"{t}.png")
                )
                if cam == "camera4":
                    # import pdb; pdb.set_trace();
                    d = np.expand_dims(d, axis=-1)
                depths.append(d)

            # import pdb; pdb.set_trace()
            arr = np.stack(depths, axis=0)
            # import pdb; pdb.set_trace();
            ################ ADDING TO REDUCE DATALOADER LATENCY ######################
            # arr = np.expand_dims(arr.mean(axis=3), axis=3)
            # arr = (torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0))
            arr = arr.astype(np.float32) / 255.0
            # import pdb; pdb.set_trace();
            arr = np.expand_dims(arr.mean(axis=3), axis=3)
            arr = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            arr = F.interpolate(
                arr,          # (1, 3, H, W)
                size=(128, 128),
                mode="bilinear",
                align_corners=False
            )
            arr = arr.numpy()
            depth_grp.create_dataset(
                cam,
                data=arr,
                chunks=(8, *arr.shape[1:]),
                compressor=zarr.Blosc(cname="lz4", clevel=3)
            )

        # ---- Heatmaps ----
        hm_grp = ep_out.create_group("heatmaps")
        for cam in ["camera1", "camera2", "camera3"]:
            hms = []
            for t in range(T):
                hm = imageio.imread(
                    os.path.join(ROOT, "unnormalized_heatmap_images", ep, cam, f"{t}.png")
                )
                hms.append(hm)
            arr = np.stack(hms, axis=0)
            # import pdb; pdb.set_trace();
            ################ ADDING TO REDUCE DATALOADER LATENCY ######################
            # arr = np.expand_dims(arr.mean(axis=3), axis=3)
            # arr = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            arr = arr.astype(np.float32) / 255.0
            arr = np.expand_dims(arr.mean(axis=3), axis=3)
            arr = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            arr = F.interpolate(
                arr,          # (1, 3, H, W)
                size=(128, 128),
                mode="bilinear",
                align_corners=False
            )
            arr = arr.numpy()
            hm_grp.create_dataset(
                cam,
                data=arr,
                chunks=(8, *arr.shape[1:]),
                compressor=zarr.Blosc(cname="lz4", clevel=3)
            )

        # ---- Gripper pose ----
        # from rvt.mvt.utils import ForkedPdb;
        # ForkedPdb().set_trace()
        # poses = []
        # for t in range(T):
        #     with open(os.path.join(ROOT, "gripper_pose", ep, f"{t}.pkl"), "rb") as f:
        #         poses.append(pickle.load(f))
        # poses = np.stack(poses, axis=0)

        # ep_out.create_dataset(
        #     "gripper_pose/pose",
        #     data=poses,
        #     chunks=(8, poses.shape[1]),
        #     compressor=zarr.Blosc(cname="lz4", clevel=3)
        # )

        gripper_grp = ep_out.create_group("gripper")

        keys = [
            "gripper_pose",
            "gripper_open_close",
            "gripper_action",
            "gripper_idx_delta_action",
        ]

        buffers = {k: [] for k in keys}

        for t in range(T):
            with open(os.path.join(ROOT, "gripper_pose", ep, f"{t}.pkl"), "rb") as f:
                data = pickle.load(f)
            # import pdb; pdb.set_trace();
            for k in keys:
                v = data[k]

                # Torch → NumPy
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()

                # ONLY fix the 1D case: (D,) -> (1, D)
                if v.ndim == 1:
                    v = v[None, :]

                buffers[k].append(v)

        # Stack and save
        for k, vals in buffers.items():
            arr = np.stack(vals, axis=0)  # (T, 1, D)

            gripper_grp.create_dataset(
                k.replace("gripper_", ""),
                data=arr.astype(np.float32),
                chunks=(min(8, arr.shape[0]), 1, arr.shape[2]),
                compressor=zarr.Blosc(cname="lz4", clevel=3),
            )

        # Sort numerically by extracting the episode number
        # replay_folders.sort(key=lambda x: int(re.search(r'episode(\d+)$', os.path.basename(x)).group(1)))
        language_grp = ep_out.create_group("language")
        replay_folder = os.path.join(folder, ep)
        # import pdb; pdb.set_trace();
        with open(replay_folder+"/variation_descriptions.pkl", "rb") as f:
                d = pickle.load(f)
        # import pdb; pdb.set_trace();
        device = torch.device("cuda")
        clip_model, _ = clip.load("RN50", device=device)  # CLIP-ResNet50
        clip_model = clip_model.to(device=device)
        clip_model.eval()
        description = d[0]
        print(description)
        # description = "Hi. My name is Pratik. Let me see what happens here. I am checking the CLIP Embeddings"
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device=device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        # import pdb; pdb.set_trace();
        language_grp.create_dataset(
                "text_embedding",
                data=np.array(lang_embs.cpu()).astype(np.float32),
                chunks=(1, *lang_embs.shape[1:]),
                compressor=zarr.Blosc(cname="lz4", clevel=3),
        )
        language_grp.create_dataset(
                "pooled_text_features",
                data=np.array(lang_feats.cpu()).astype(np.float32),
                chunks=(1, *lang_feats.shape[1:]),
                compressor=zarr.Blosc(cname="lz4", clevel=3),
        )
        

