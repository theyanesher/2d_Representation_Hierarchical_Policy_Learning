"""Dataset adapter for pre-processed .npz files (Coffee Task format)."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from lfd3d.datasets.base_data import BaseDataModule, BaseDataset
from lfd3d.datasets.rgb_text_feature_gen import get_siglip_text_embedding
from lfd3d.utils.data_utils import collate_pcd_fn
from transformers import AutoModel, AutoProcessor


def _load_or_compute_text_embed(
    caption: str, cache_path: str | None, use_text_embed: bool
) -> np.ndarray:
    """
    Return a text embedding for the given caption.
    - use_text_embed=False: return zeros (single-task training; no SigLIP loaded)
    - use_text_embed=True:  load from cache_path if it exists, otherwise run
      SigLIP once and save the result to cache_path for future runs.
    """
    siglip_dim = 1152  # google/siglip-so400m-patch14-384 text output dim

    if not use_text_embed:
        return np.zeros(siglip_dim, dtype=np.float32)

    if cache_path and Path(cache_path).exists():
        return np.load(cache_path).astype(np.float32)

    siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    embed = get_siglip_text_embedding(
        caption, siglip=siglip, siglip_processor=processor, device="cpu"
    )  # (1152,) float32

    if cache_path:
        np.save(cache_path, embed)

    return embed


class NpyDataset(BaseDataset):
    """
    Dataset for per-frame .npz files produced by the Coffee Task data pipeline.

    Expected directory layout:
        data_dir/
            demo_0/
                0.npz
                1.npz
                ...
            demo_1/
                ...

    Each .npz contains:
        point_cloud      (1, N, 3)   scene point cloud
        gripper_pcd      (1, 4, 3)   gripper keypoints at this frame
        goal_gripper_pcd (1, 4, 3)   gripper keypoints at the goal frame
        rgb_agentview    (1, H, W, 3)
        depth_agentview  (1, H, W, 1)
        agentview_intrinsics (1, 3, 3)
        agentview_extrinsics (1, 4, 4)
        ...
    """

    GRIPPER_IDX = np.array([0, 1, 2])

    def __init__(
        self,
        dataset_cfg,
        split: str = "train",
        split_indices=None,
        augment_train=None,
        augment_cfg=None,
    ):
        super().__init__(augment_train=augment_train, augment_cfg=augment_cfg)
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.data_dir = Path(dataset_cfg.data_dir)
        self.task_caption = dataset_cfg.task_caption

        # Collect all frame paths, sorted deterministically
        all_frames = []
        for demo_dir in sorted(self.data_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            frames = sorted(demo_dir.glob("*.npz"), key=lambda p: int(p.stem))
            all_frames.extend(frames)

        if split_indices is not None:
            self.frames = [all_frames[i] for i in split_indices]
        else:
            self.frames = all_frames

        # Compute (or load from cache) the text embedding once in the main process.
        # Workers receive it via pickle — no SigLIP reload per worker.
        cache_path = dataset_cfg.get("text_embed_cache", None)
        use_text_embed = dataset_cfg.get("use_text_embed", False)
        self.text_embed = _load_or_compute_text_embed(
            self.task_caption, cache_path, use_text_embed
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        d = np.load(self.frames[idx], allow_pickle=True)

        anchor_pcd = d["point_cloud"][0].astype(np.float32)       # (N, 3)
        action_pcd = d["gripper_pcd"][0].astype(np.float32)       # (4, 3)
        goal_pcd   = d["goal_gripper_pcd"][0].astype(np.float32)  # (4, 3)

        # Normalization: follow base_data.get_normalize_mean_std
        # With normalize=False (default) this is identity (mean=0, std=1)
        if self.dataset_cfg.get("normalize", False):
            pcd_mean = action_pcd.mean(axis=0)
            pcd_std  = anchor_pcd.std(axis=0)
        else:
            pcd_mean = np.zeros(3, dtype=np.float32)
            pcd_std  = np.ones(3, dtype=np.float32)

        action_pcd_norm = (action_pcd - pcd_mean) / pcd_std
        anchor_pcd_norm = (anchor_pcd - pcd_mean) / pcd_std
        cross_displacement = (goal_pcd - action_pcd) / pcd_std  # (4, 3)

        # Dummy features: model won't read them with use_rgb=False, but
        # collate_pcd_fn always expects anchor_feat_pcd alongside anchor_pcd.
        anchor_feat_pcd = np.zeros(
            (anchor_pcd_norm.shape[0], 3), dtype=np.float32
        )

        # Camera data (primary: agentview)
        rgb   = d["rgb_agentview"][0].astype(np.uint8)            # (H, W, 3)
        depth = d["depth_agentview"][0, :, :, 0].astype(np.float32)  # (H, W)
        K     = d["agentview_intrinsics"][0].astype(np.float32)   # (3, 3)
        T     = d["agentview_extrinsics"][0].astype(np.float32)   # (4, 4)

        # Stack start/end frames; data is already a single frame so duplicate
        rgbs   = np.stack([rgb, rgb], axis=0)    # (2, H, W, 3)
        depths = np.stack([depth, depth], axis=0)  # (2, H, W)

        # Dummy gripper trajectory (10 future steps)
        gripper_trajectory = np.tile(action_pcd_norm[[0]], (10, 1)).astype(np.float32)

        return {
            # Primary camera
            "rgbs":       rgbs,
            "depths":     depths,
            "intrinsics": K,
            "extrinsics": T,
            # Auxiliary cameras (none)
            "aux_rgbs":       np.zeros((0, 2, rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8),
            "aux_depths":     np.zeros((0, 2, rgb.shape[0], rgb.shape[1]), dtype=np.float32),
            "aux_intrinsics": np.zeros((0, 3, 3), dtype=np.float32),
            "aux_extrinsics": np.zeros((0, 4, 4), dtype=np.float32),
            # Point clouds
            "action_pcd":       action_pcd_norm,   # (4, 3) → Pointclouds
            "anchor_pcd":       anchor_pcd_norm,   # (N, 3) → Pointclouds(features=anchor_feat_pcd)
            "anchor_feat_pcd":  anchor_feat_pcd,   # (N, 3) zeros
            "gripper_trajectory": gripper_trajectory,  # (10, 3) → Pointclouds
            # Labels
            "cross_displacement": cross_displacement,  # (4, 3) → Pointclouds
            # Text
            "caption":        self.task_caption,
            "text_embed":     self.text_embed,
            "actual_caption": self.task_caption,
            # Transforms / normalization
            "start2end":  np.eye(4, dtype=np.float32),
            "pcd_mean":   pcd_mean,
            "pcd_std":    pcd_std,
            "augment_R":  np.eye(3, dtype=np.float32),
            "augment_t":  np.zeros(3, dtype=np.float32),
            "augment_C":  anchor_pcd.mean(axis=0).astype(np.float32),
            # Metadata
            "gripper_idx": self.GRIPPER_IDX,
            "vid_name":    str(self.frames[idx]),
            "data_source": "libero_franka",
        }


class NpyDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size,
        val_batch_size,
        num_workers,
        dataset_cfg,
        seed,
        augment_train=None,
        augment_cfg=None,
    ):
        super().__init__(
            batch_size,
            val_batch_size,
            num_workers,
            dataset_cfg,
            seed,
            augment_train,
            augment_cfg,
        )

    def _generate_splits(self):
        """Split by demo directory: 90% train, 10% val."""
        data_dir = Path(self.dataset_cfg.data_dir)
        demo_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

        rng = random.Random(self.seed)
        demo_dirs_shuffled = demo_dirs[:]
        rng.shuffle(demo_dirs_shuffled)

        n_val = max(1, int(len(demo_dirs_shuffled) * self.dataset_cfg.val_episode_ratio))
        val_demos  = set(str(d) for d in demo_dirs_shuffled[:n_val])
        train_demos = set(str(d) for d in demo_dirs_shuffled[n_val:])

        # Build frame-level indices relative to the full sorted frame list
        all_frames = []
        for demo_dir in sorted(data_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            frames = sorted(demo_dir.glob("*.npz"), key=lambda p: int(p.stem))
            all_frames.extend(frames)

        train_indices = [i for i, f in enumerate(all_frames) if str(f.parent) in train_demos]
        val_indices   = [i for i, f in enumerate(all_frames) if str(f.parent) in val_demos]
        return train_indices, val_indices

    def setup(self, stage: str = "fit"):
        self.stage = stage
        train_indices, val_indices = self._generate_splits()

        self.train_dataset = NpyDataset(
            dataset_cfg=self.dataset_cfg,
            split="train",
            split_indices=train_indices,
            augment_train=self.augment_train,
            augment_cfg=self.augment_cfg,
        )
        self.val_datasets = {
            "coffee_task": NpyDataset(
                dataset_cfg=self.dataset_cfg,
                split="val",
                split_indices=val_indices,
                augment_train=None,
                augment_cfg=self.augment_cfg,
            )
        }
        self.val_tags = list(self.val_datasets.keys())
        self.test_datasets = {}
