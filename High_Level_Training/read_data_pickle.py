import os
import zarr
import numpy as np
import imageio
import pickle
from tqdm import tqdm
import torch
import random

path = "data_diffusion_policy/outputs_1st_insert_onto_square_peg/trajectories_data/episode0.zarr"
ep = zarr.open(path, mode="r")
T = 100
t0 = random.randint(0, T - 4)
from rvt.mvt.utils import ForkedPdb;
ForkedPdb().set_trace()
rgb = ep["rgb/camera1"][t0:t0+4]
depth = ep["depth/camera1"][t0:t0+4]
action = ep["gripper/action"][t0:t0+4]
idx_delta_action = ep["gripper/idx_delta_action"][t0:t0+4]
open_close = ep["gripper/open_close"][t0:t0+4]



# episodes = sorted(os.listdir(os.path.join(ROOT, "unnormalized_rgb")))

# for ep in tqdm(episodes):
#     ep_out = zarr.open(
#         os.path.join(OUT, f"{ep}.zarr"),
#         mode="w"
#     )

#     # Infer timesteps
#     cam1_dir = os.path.join(ROOT, "unnormalized_rgb", ep, "camera1")
#     timesteps = sorted(os.listdir(cam1_dir))
#     T = len(timesteps)

#     # ---- RGB ----
#     rgb_grp = ep_out.create_group("rgb")
#     for cam in ["camera1", "camera2", "camera3"]:
#         imgs = []
#         for t in range(T):
#             img = imageio.imread(
#                 os.path.join(ROOT, "unnormalized_rgb", ep, cam, f"{t}.png")
#             )
#             imgs.append(img)
#         arr = np.stack(imgs, axis=0)  # (T, H, W, 3)

#         rgb_grp.create_dataset(
#             cam,
#             data=arr,
#             chunks=(8, *arr.shape[1:]),
#             compressor=zarr.Blosc(cname="lz4", clevel=3)
#         )

#     # ---- Depth ----
#     depth_grp = ep_out.create_group("depth")
#     for cam in ["camera1", "camera2", "camera3"]:
#         depths = []
#         for t in range(T):
#             d = imageio.imread(
#                 os.path.join(ROOT, "depth", ep, cam, f"{t}.png")
#             )
#             depths.append(d)
#         arr = np.stack(depths, axis=0)

#         depth_grp.create_dataset(
#             cam,
#             data=arr,
#             chunks=(8, *arr.shape[1:]),
#             compressor=zarr.Blosc(cname="lz4", clevel=3)
#         )

#     # ---- Heatmaps ----
#     hm_grp = ep_out.create_group("heatmaps")
#     for cam in ["camera1", "camera2", "camera3"]:
#         hms = []
#         for t in range(T):
#             hm = imageio.imread(
#                 os.path.join(ROOT, "unnormalized_heatmap_images", ep, cam, f"{t}.png")
#             )
#             hms.append(hm)
#         arr = np.stack(hms, axis=0)

#         hm_grp.create_dataset(
#             cam,
#             data=arr,
#             chunks=(8, *arr.shape[1:]),
#             compressor=zarr.Blosc(cname="lz4", clevel=3)
#         )

#     # ---- Gripper pose ----
#     # from rvt.mvt.utils import ForkedPdb;
#     # ForkedPdb().set_trace()
#     # poses = []
#     # for t in range(T):
#     #     with open(os.path.join(ROOT, "gripper_pose", ep, f"{t}.pkl"), "rb") as f:
#     #         poses.append(pickle.load(f))
#     # poses = np.stack(poses, axis=0)

#     # ep_out.create_dataset(
#     #     "gripper_pose/pose",
#     #     data=poses,
#     #     chunks=(8, poses.shape[1]),
#     #     compressor=zarr.Blosc(cname="lz4", clevel=3)
#     # )
#     gripper_grp = ep_out.create_group("gripper")

#     keys = [
#         "gripper_pose",
#         "gripper_open_close",
#         "gripper_action",
#         "gripper_idx_delta_action",
#     ]

#     buffers = {k: [] for k in keys}

#     for t in range(T):
#         with open(os.path.join(ROOT, "gripper_pose", ep, f"{t}.pkl"), "rb") as f:
#             data = pickle.load(f)

#         for k in keys:
#             v = data[k]

#             # Torch → NumPy
#             if isinstance(v, torch.Tensor):
#                 v = v.detach().cpu().numpy()

#             # ONLY fix the 1D case: (D,) -> (1, D)
#             if v.ndim == 1:
#                 v = v[None, :]

#             buffers[k].append(v)

#     # Stack and save
#     for k, vals in buffers.items():
#         arr = np.stack(vals, axis=0)  # (T, 1, D)

#         gripper_grp.create_dataset(
#             k.replace("gripper_", ""),
#             data=arr.astype(np.float32),
#             chunks=(min(8, arr.shape[0]), 1, arr.shape[2]),
#             compressor=zarr.Blosc(cname="lz4", clevel=3),
#         )
