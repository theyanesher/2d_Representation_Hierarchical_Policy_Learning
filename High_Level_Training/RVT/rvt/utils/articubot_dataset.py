"""
Articubot HDF5 Dataset loader for RVT training.

Each HDF5 file is one episode. Under the file root:
    obs/
        cam0_image   (T, H, W, C)  uint8    - RGB camera 0
        cam1_image   (T, H, W, C)  uint8    - RGB camera 1
        cam2_image   (T, H, W, C)  uint8    - RGB camera 2
        cam0_depth   (T, H, W)     uint16   - depth camera 0 (mm)
        cam1_depth   (T, H, W)     uint16   - depth camera 1 (mm)
        cam0_heatmap (T, H, W)     float16  - heatmap camera 0
        cam1_heatmap (T, H, W)     float16  - heatmap camera 1
        state        (T, D)        float32  - robot proprioception
        ...any other keys present in the files
    action/
        hybrid       (T, A)        float32  - hybrid delta action

The __getitem__ returns a flat dict of tensors keyed by the original obs key name
(e.g. "cam0_image", "state") plus "action_hybrid" (or whatever action keys exist).

RGB images    -> (C, H, W) float32 in [0, 1]
Depth images  -> (1, H, W) float32 in metres  (divide uint16 mm values by 1000)
Heatmaps      -> (1, H, W) float32
Low-dim data  -> (D,)      float32
Actions       -> (A,)      float32
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class ArticubotH5Dataset(Dataset):
    """Per-timestep dataset over Articubot HDF5 demonstration files.

    Args:
        data_dir: Directory containing ``*.h5`` episode files.
        max_episodes: If given, only use the first N episode files.
    """

    def __init__(self, data_dir: str, max_episodes: int = None):
        self.data_dir = Path(data_dir)
        self.h5_paths = sorted(self.data_dir.glob("*.h5"))
        if not self.h5_paths:
            raise ValueError(f"No .h5 files found in {data_dir}")
        if max_episodes is not None:
            self.h5_paths = self.h5_paths[:max_episodes]

        # Discover available keys from the first file.
        with h5py.File(str(self.h5_paths[0]), "r") as f:
            self.obs_keys = list(f["obs"].keys())
            self.action_keys = list(f["action"].keys()) if "action" in f else []

        # Build a flat index: list of (file_path_str, timestep_idx)
        self.index = []
        for path in self.h5_paths:
            with h5py.File(str(path), "r") as f:
                T = f[f"obs/{self.obs_keys[0]}"].shape[0]
            for t in range(T):
                self.index.append((str(path), t))

        print(
            f"[ArticubotH5Dataset] {len(self.h5_paths)} episodes, "
            f"{len(self.index)} timesteps total.\n"
            f"  obs keys:    {self.obs_keys}\n"
            f"  action keys: {self.action_keys}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        path, t = self.index[idx]
        sample = {}

        with h5py.File(path, "r") as f:
            for key in self.obs_keys:
                raw = f[f"obs/{key}"][t]
                sample[key] = _to_tensor(key, raw)

            for key in self.action_keys:
                raw = f[f"action/{key}"][t]
                arr = np.asarray(raw, dtype=np.float32)
                sample[f"action_{key}"] = torch.from_numpy(arr)

        return sample


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def _to_tensor(key: str, data) -> torch.Tensor:
    """Convert a raw numpy array (one timestep) to a properly typed tensor."""
    data = np.asarray(data)

    if "image" in key:
        # (H, W, C) uint8 -> (C, H, W) float32 normalised to [0, 1]
        arr = np.moveaxis(data, -1, 0).astype(np.float32) / 255.0

    elif "depth" in key:
        # (H, W) uint16 [mm] -> (1, H, W) float32 [m]
        arr = data.astype(np.float32) / 1000.0
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # add channel dim -> (1, H, W)

    elif "heatmap_ghost" in key:
        # (H, W, 4) uint8 -> (4, H, W) float32 normalised to [0, 1]
        arr = np.moveaxis(data, -1, 0).astype(np.float32) / 255.0

    elif "heatmap" in key:
        # (H, W) float16 -> (1, H, W) float32
        arr = data.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

    else:
        # state, intrinsics, extrinsics, etc. -> flat float32
        arr = data.astype(np.float32)

    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_articubot_dataset(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    max_episodes: int = None,
) -> DataLoader:
    """Create a PyTorch DataLoader over the Articubot HDF5 dataset.

    Args:
        data_dir: Path to the directory containing ``*.h5`` episode files,
                  e.g. ``/scratch/.../Heatmap_Articubot_Dataset/``.
        batch_size: Number of timesteps per batch.
        num_workers: Number of DataLoader worker processes.
        max_episodes: Optional cap on the number of episode files to use.

    Returns:
        A ``torch.utils.data.DataLoader`` that yields dicts of tensors with
        keys like ``"cam0_image"``, ``"cam1_image"``, ``"cam2_image"``,
        ``"state"``, ``"cam0_depth"``, ``"cam0_heatmap"``, ``"action_hybrid"``,
        etc.  (Exactly the keys present in the HDF5 files.)
    """
    dataset = ArticubotH5Dataset(data_dir, max_episodes=max_episodes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader
