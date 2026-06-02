"""
GHOST high-level goal predictor for the hierarchical RLBench eval.

GHOST is the lfd3d `ArticubotNetwork` (PointNet++ GMM over scene anchors with
optional FiLM text conditioning). It predicts a 4-point goal gripper point
cloud directly in WORLD frame, which is exactly what the DP3 low-level consumes
as `goal_gripper_pcd`.

Integration follows the proven MimicGen pattern
(SMITH_MimicGen/.../eval_ghost_high_level_2d_dit_low_level.py): we add the
lfd3d repo to sys.path and stub the heavy training-only deps
(pytorch_lightning, wandb, diffusers, pytorch3d.{structures,ops}, trimesh,
imageio, transformers) plus three lfd3d sub-modules (dino_heatmap, tax3d,
viz_utils) in sys.modules so `import lfd3d.models.articubot` succeeds without
installing those packages into the eval env. ArticubotNetwork's forward never
touches any of them.

Scene preprocessing mirrors the GHOST training data exactly
(SMITH_High_Level_FineTune/rlbench_to_d2_converter.py):
  fuse 4 cams [front, left_shoulder, right_shoulder, wrist] (world frame)
  -> crop to SCENE_BOUNDS -> FPS to 4500 points.
The current gripper 4-pt pcd is derived from the current gripper pose+open via
the same RVT/DP3 template (dp3_agent.build_gripper_pcd_from_pose), and the text
embedding is zeros(1152) (the ckpt was trained text-off).
"""

import os
import sys
import types
import importlib

import numpy as np
import torch

from rvt.utils.peract_utils import SCENE_BOUNDS
from rvt.models.dp3_agent import dp3_agent


# GHOST training scene cloud size (rlbench_to_d2_converter.py:N_POINTS).
GHOST_NUM_POINTS = 4500
# Camera fusion order must match the converter (rlbench_to_d2_converter.py:CAMERAS)
# so the FPS seed (index 0) lands on the same camera as at training time.
GHOST_CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]

DEFAULT_LFD3D_REPO = (
    "/project_data/held/pratik/run_sample_basic_experiments/Bimanual_Manipulation/"
    "Articubot_Data_For_RVT/lfd3d/lfd3d"
)


# -----------------------------------------------------------------------------
# sys.modules stubbing (ported verbatim from the MimicGen GHOST eval).
# -----------------------------------------------------------------------------
def _ensure_stub(name, attrs=None):
    """Try to import; if missing, install a minimal stub. Never overrides a real module."""
    try:
        __import__(name)
        return
    except ImportError:
        pass
    parts = name.split(".")
    for i in range(len(parts)):
        sub = ".".join(parts[: i + 1])
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)
        if i > 0:
            setattr(sys.modules[".".join(parts[:i])], parts[i], sys.modules[sub])
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)


def _force_stub_module(name, attrs=None):
    """Force a stub into sys.modules even if a real module exists. Use only for
    sub-modules whose attributes we know are never reached on the inference path."""
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    parent_name, _, leaf = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], leaf, m)


def _stub_unused_lfd3d_deps():
    """Stub training-time deps that lfd3d.models.articubot loads at module level
    but ArticubotNetwork's forward never reaches."""
    _ensure_stub(
        "pytorch_lightning",
        {"LightningModule": type("LightningModule", (), {})},
    )
    _ensure_stub(
        "diffusers",
        {"get_cosine_schedule_with_warmup": lambda *a, **k: None},
    )
    _ensure_stub("pytorch3d")
    _ensure_stub("pytorch3d.structures", {"Pointclouds": type("Pointclouds", (), {})})
    _ensure_stub("pytorch3d.ops", {"sample_farthest_points": lambda *a, **k: None})
    _ensure_stub("wandb")
    _ensure_stub("trimesh")
    _ensure_stub("imageio")
    _ensure_stub("imageio.v3")
    _ensure_stub(
        "transformers",
        {
            "AutoImageProcessor": type("AutoImageProcessor", (), {}),
            "AutoModel": type("AutoModel", (), {}),
        },
    )

    # Pre-load the real (empty) lfd3d parent packages so sub-module stubs attach.
    importlib.import_module("lfd3d")
    importlib.import_module("lfd3d.models")

    _noop = lambda *a, **k: None
    _force_stub_module("lfd3d.models.dino_heatmap", {"calc_pix_metrics": _noop})
    _force_stub_module("lfd3d.models.tax3d", {"calc_pcd_metrics": _noop})
    _force_stub_module("lfd3d.utils")
    _force_stub_module(
        "lfd3d.utils.viz_utils",
        {
            "get_action_anchor_pcd": _noop,
            "get_img_and_track_pcd": _noop,
            "invert_augmentation_and_normalization": _noop,
            "project_pcd_on_image": _noop,
        },
    )


def _build_articubot_model_cfg(in_channels=4, use_rgb=False):
    """Mirror run_gmm_on_dataset.py:build_model_cfg / the MimicGen eval."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "name": "articubot",
            "type": "cross_displacement",
            "in_channels": in_channels,
            "num_classes": 13,
            "keep_gripper_in_fps": False,
            "add_action_pcd_masked": True,
            "use_text_embedding": True,
            "use_rgb": use_rgb,
            "use_dual_head": False,
            "is_gmm": True,
            "fixed_variance": [0.01, 0.05, 0.1, 0.25, 0.5],
            "uniform_weights_coeff": 0.1,
        }
    )


class GhostHighLevel:
    """High-level goal predictor wrapping lfd3d ArticubotNetwork.

    `predict_goal_pcd(observation)` returns the (1, 4, 3) world-frame goal
    gripper point cloud to feed DP3 as `goal_gripper_pcd`.
    """

    def __init__(
        self,
        ckpt_path,
        device,
        lfd3d_repo=DEFAULT_LFD3D_REPO,
        in_channels=4,
        use_rgb=False,
        num_points=GHOST_NUM_POINTS,
        argmax_weight=True,
        cameras=GHOST_CAMERAS,
    ):
        self.device = device
        self.num_points = num_points
        self.argmax_weight = argmax_weight
        self.cameras = list(cameras)
        self.scene_bounds = list(SCENE_BOUNDS)

        # Make lfd3d importable, then stub heavy deps before importing the model.
        src = os.path.join(lfd3d_repo, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        _stub_unused_lfd3d_deps()

        from lfd3d.models.articubot import ArticubotNetwork

        model_cfg = _build_articubot_model_cfg(in_channels=in_channels, use_rgb=use_rgb)
        network = ArticubotNetwork(model_cfg=model_cfg).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        network_state = {
            k[len("network."):]: v for k, v in state.items() if k.startswith("network.")
        }
        missing, unexpected = network.load_state_dict(network_state, strict=False)
        if missing or unexpected:
            print(f"[GHOST] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        network.eval()
        self.network = network

        # Reused for the current-gripper 4-pt template (identical to GHOST training).
        self.gripper_template_4 = dp3_agent.make_gripper_template_4(
            device=torch.device(device), dtype=torch.float32
        )
        print(f"[GHOST] loaded ArticubotNetwork from {ckpt_path}")

    def eval(self):
        self.network.eval()

    def train(self):
        # GHOST is inference-only here.
        self.network.eval()

    # -------------------------------------------------------------------------
    def _build_scene_pcd(self, observation):
        """Fuse 4 cams (world frame) -> crop SCENE_BOUNDS -> FPS to num_points.
        Mirrors rlbench_to_d2_converter.build_camera_arrays exactly."""
        device = self.device
        cam_points = []
        for cam in self.cameras:
            p = observation[f"{cam}_point_cloud"].to(device=device, dtype=torch.float32)
            if p.dim() == 5 and p.shape[1] == 1:
                p = p[:, 0]            # (B,3,H,W)
            elif p.dim() == 4:
                pass
            else:
                raise RuntimeError(f"{cam}_point_cloud unexpected shape: {p.shape}")
            B, C, H, W = p.shape
            p = p.permute(0, 2, 3, 1).reshape(B, H * W, 3)
            cam_points.append(p)

        scene = torch.cat(cam_points, dim=1)   # (B, N, 3), front-first order

        # SCENE_BOUNDS crop (per batch element).
        x0, y0, z0, x1, y1, z1 = self.scene_bounds
        B = scene.shape[0]
        out = []
        for b in range(B):
            pc = scene[b]
            m = (
                (pc[:, 0] >= x0) & (pc[:, 0] <= x1)
                & (pc[:, 1] >= y0) & (pc[:, 1] <= y1)
                & (pc[:, 2] >= z0) & (pc[:, 2] <= z1)
            )
            cropped = pc[m]
            if cropped.shape[0] == 0:
                cropped = pc
            if cropped.shape[0] >= self.num_points:
                idx = dp3_agent.farthest_point_sampling(
                    cropped.unsqueeze(0), self.num_points
                )[0]
                sampled = cropped[idx]
            else:
                pad = torch.randint(
                    0, cropped.shape[0], (self.num_points - cropped.shape[0],), device=device
                )
                sampled = torch.cat([cropped, cropped[pad]], dim=0)
            out.append(sampled)
        return torch.stack(out, dim=0)   # (B, num_points, 3)

    def _build_current_gripper_pcd(self, observation):
        """Current gripper 4-pt pcd from gripper_pose+open via the DP3 template."""
        device = self.device
        gp = observation["gripper_pose"].to(device=device, dtype=torch.float32)
        go = observation["gripper_open"].to(device=device, dtype=torch.float32)
        if gp.dim() == 3 and gp.shape[1] == 1:
            gp = gp[:, 0]
        if go.dim() == 3 and go.shape[1] == 1:
            go = go[:, 0]
        # canonicalize quaternion sign (qw >= 0), as the converter does.
        quat = gp[:, 3:7]
        mask = quat[:, 3] < 0
        gp[mask, 3:7] = -gp[mask, 3:7]
        return dp3_agent.build_gripper_pcd_from_pose(gp, go, self.gripper_template_4)

    @torch.no_grad()
    def predict_goal_pcd(self, observation):
        """Returns (1, 4, 3) world-frame goal gripper point cloud."""
        scene_pcd = self._build_scene_pcd(observation)            # (B, N, 3)
        gripper_pcd = self._build_current_gripper_pcd(observation)  # (B, 4, 3)
        B = scene_pcd.shape[0]
        text_embed = torch.zeros(B, 1152, device=self.device, dtype=torch.float32)
        return infer_articubot_subgoal(
            self.network, scene_pcd, gripper_pcd, text_embed,
            argmax_weight=self.argmax_weight,
        )


@torch.no_grad()
def infer_articubot_subgoal(network, scene_pcd_t, gripper_pcd_t, text_embed_t, argmax_weight=True):
    """Mirror run_gmm_on_dataset.py:infer_gmm. Returns (B, 4, 3) world-frame xyz.

    gripper-FIRST concat with a 1/0 mask channel (prepare_scene_pcd with
    add_action_pcd_masked=True, use_rgb=False); softmax weights over scene
    anchors only (gripper anchors stripped); argmax/multinomial pick; goal =
    anchor + displacement.
    """
    device = scene_pcd_t.device
    B, K, _ = gripper_pcd_t.shape
    N = scene_pcd_t.shape[1]

    gripper_w_mask = torch.cat([gripper_pcd_t, torch.ones(B, K, 1, device=device)], dim=2)
    scene_w_mask = torch.cat([scene_pcd_t, torch.zeros(B, N, 1, device=device)], dim=2)
    full = torch.cat([gripper_w_mask, scene_w_mask], dim=1)        # (B, K+N, 4)
    anchor_xyz_full = full[:, :, :3].clone()
    net_in = full.permute(0, 2, 1).contiguous()                    # (B, 4, K+N)

    outputs = network(net_in, text_embedding=text_embed_t, data_source=["libero_franka"] * B)

    B, KN, _ = outputs.shape
    weights = outputs[:, :, -1]                                    # (B, K+N)
    displacements = outputs[:, :, :-1].reshape(B, KN, 4, 3)        # (B, K+N, 4, 3)
    gaussian_means = anchor_xyz_full[:, :, None, :] + displacements

    weights = weights[:, K:]
    gaussian_means = gaussian_means[:, K:, :, :]

    probabilities = torch.nn.functional.softmax(weights, dim=1)
    if argmax_weight:
        sampled_idx = torch.argmax(probabilities, dim=1)
    else:
        sampled_idx = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    batch_idx = torch.arange(B, device=outputs.device)
    return gaussian_means[batch_idx, sampled_idx]                  # (B, 4, 3)
