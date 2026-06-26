"""
Batch-optimized variant of run_gmm_on_dataset.py.

Same outputs as run_gmm_on_dataset.py — same H5 schema (obs/gmm_pred_goal,
obs/gmm_all_goals, obs/gmm_all_weights, all original obs/* fields,
action/delta, action/hybrid, _physical/cam*), same model loading, same
infer_gmm math (gripper-first concat + mask channel; strip-K=4 gripper
anchors; softmax weights; argmax or multinomial sampling).

Per-demo timesteps are batched through ArticubotNetwork in chunks of
--batch_size instead of one-at-a-time. The PointNet++ FPS Python loop launches
a fixed number of CUDA kernels regardless of batch size, so per-sample cost
drops sharply once B>1.

--rl_bench adapter
------------------
With --rl_bench the dataset is treated as the RL Bench converted format:
  * 4 cameras (front, left_shoulder, right_shoulder, wrist) mapped to the
    GROOT cam0..cam3 schema — RGB stored (H,W,3) uint8, depth uint16 mm — so
    the output matches the existing NO_GMM GROOT training h5 exactly.
  * Every remaining npz key is copied verbatim into obs/<key> (eef_pos,
    eef_quat, gripper_qpos, lang_goal [vlen str], episode_idx, reward,
    terminal, timeout, the per-camera point clouds, ...).
  * The raw absolute action is also stored verbatim at obs/action (in addition
    to action/delta + action/hybrid).
  * If gripper_pcd is absent it is derived from `state` via the RVT template.
The MimicGen (default, 2-camera) path is unchanged.

Phase 1: load all T npz files for the demo into preallocated CPU arrays.
Phase 2: run the network in chunks of --batch_size.
Phase 3: (optional) viser viz, fed from CPU arrays.
Phase 4: write the consolidated H5 file.

Resume support: --start_demo skips earlier demo_* directories, and any demo
whose output .h5 already exists is skipped.

Example:
  PYTHONNOUSERSITE=1 pixi run python scripts/run_gmm_on_dataset_batch_optimized.py \\
      --rl_bench \\
      --dataset_dir /scratch/.../sweep_to_dustpan_of_size \\
      --ckpt_path   /path/to/.ckpt \\
      --gmm_output_dir /local/.../gmm_h5 \\
      --start_demo 0 --max_files 1000 --batch_size 164
"""

import argparse
import os
from collections import defaultdict

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from lfd3d.models.articubot import ArticubotNetwork


def build_model_cfg(in_channels, use_rgb):
    """Default model_cfg matching the Coffee_Task articubot training run."""
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


def load_articubot(ckpt_path, model_cfg, device="cuda"):
    network = ArticubotNetwork(model_cfg=model_cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # GoalRegressionModule wraps the network as self.network, so saved keys are
    # prefixed with "network." — strip that prefix to load directly into ArticubotNetwork.
    network_state = {
        k[len("network.") :]: v for k, v in state.items() if k.startswith("network.")
    }
    missing, unexpected = network.load_state_dict(network_state, strict=False)
    if missing or unexpected:
        print(f"[load_articubot] missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    print(f"Successfully loaded model from: {ckpt_path}")
    network.eval()
    return network


def read_demo_lang_goal(demo_dir):
    """Read the per-episode `lang_goal` string from a demo's first .npz.

    lang_goal is constant within a demo (one RL Bench variation per episode),
    so reading the first frame is enough. Used by the --use_lang_goal path to
    pick the per-demo text embedding.
    """
    npz_files = sorted(
        [f for f in os.listdir(demo_dir) if f.endswith(".npz")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    if not npz_files:
        raise ValueError(f"No .npz files in {demo_dir}")
    data = np.load(os.path.join(demo_dir, npz_files[0]), allow_pickle=True)
    if "lang_goal" not in data.files:
        raise KeyError(
            f"--use_lang_goal set but 'lang_goal' is missing from "
            f"{npz_files[0]} in {demo_dir}"
        )
    return str(np.asarray(data["lang_goal"]).reshape(-1)[0])


# ----------------------------------------------------------------------------
# RVT gripper template (copied verbatim from
# SMITH_MimicGen/.../visualize_npz_demo_RL_BENCH.py) — used to derive a
# 4-point current gripper pcd from `state` when the npz has no `gripper_pcd`.
# state = [x,y,z, qx,qy,qz,qw, gripper_open].
# ----------------------------------------------------------------------------
def quat_to_rotmat_np(q):
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def make_gripper_template_4_np():
    left_finger = np.array([0.0, -0.0405, 0.0800], dtype=np.float32)
    right_finger = np.array([0.0, 0.0405, 0.0800], dtype=np.float32)
    wrist = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    center_contact = 0.5 * (left_finger + right_finger)
    return np.stack([left_finger, right_finger, wrist, center_contact], axis=0)


def build_gripper_pcd_from_pose_np(pose_7, open_1, template_4):
    """pose_7 (N,7) [x,y,z,qx,qy,qz,qw], open_1 (N,1) in [0,1] -> (N,4,3) world."""
    t = pose_7[:, :3]
    R = quat_to_rotmat_np(pose_7[:, 3:7])
    p = np.repeat(template_4[None, :, :], repeats=pose_7.shape[0], axis=0).copy()
    delta = 0.021 * (np.clip(open_1[:, 0], 0.0, 1.0) - 0.5)
    p[:, 0, 1] -= delta
    p[:, 1, 1] += delta
    p[:, 3, :] = 0.5 * (p[:, 0, :] + p[:, 1, :])
    p_world = np.einsum("bij,bpj->bpi", R, p) + t[:, None, :]
    return p_world.astype(np.float32)


_GRIPPER_TEMPLATE_4 = make_gripper_template_4_np()


def prepare_network_input(scene_pcd, gripper_pcd):
    """
    Build the (B, C, K+N) input expected by ArticubotNetwork.

    Replicates GoalRegressionModule.prepare_scene_pcd for:
        add_action_pcd_masked=True, use_rgb=False
    Gripper-first concatenation matches the codebase.

    Args:
        scene_pcd:   (B, N, 3) tensor
        gripper_pcd: (B, K, 3) tensor (K=4)
    Returns:
        net_in:        (B, 4, K+N) input for ArticubotNetwork.forward
        anchor_pcd_xyz: (B, K+N, 3) the xyz of every anchor (gripper + scene), in order
    """
    device = scene_pcd.device
    B, K, _ = gripper_pcd.shape
    N = scene_pcd.shape[1]

    gripper_w_mask = torch.cat([gripper_pcd, torch.ones(B, K, 1, device=device)], dim=2)
    scene_w_mask = torch.cat([scene_pcd, torch.zeros(B, N, 1, device=device)], dim=2)
    full = torch.cat([gripper_w_mask, scene_w_mask], dim=1)  # (B, K+N, 4)

    anchor_pcd_xyz = full[:, :, :3].clone()
    net_in = full.permute(0, 2, 1).contiguous()  # (B, 4, K+N)
    return net_in, anchor_pcd_xyz


@torch.no_grad()
def infer_gmm(network, scene_pcd, gripper_pcd, text_embed, args):
    """
    Args:
        scene_pcd:   (B, N, 3) tensor on cuda
        gripper_pcd: (B, K, 3) tensor on cuda (K=4)
        text_embed:  (B, 1152) tensor on cuda
    Returns:
        prediction:     (B, 4, 3)        sampled goal gripper points
        probabilities:  (B, N)           softmaxed weights over scene anchors only
        anchor_points:  (B, N, 3)        scene anchor xyz (gripper points stripped)
        gmm_components: (B, N, 4, 3)     per-anchor goal predictions (anchor + displacement)
    """
    K = gripper_pcd.shape[1]
    net_in, anchor_xyz_full = prepare_network_input(scene_pcd, gripper_pcd)

    outputs = network(
        net_in, text_embedding=text_embed, data_source=["libero_franka"] * scene_pcd.shape[0]
    )  # (B, K+N, 13)

    B, KN, _ = outputs.shape
    weights = outputs[:, :, -1]                                      # (B, K+N)
    displacements = outputs[:, :, :-1].reshape(B, KN, 4, 3)          # (B, K+N, 4, 3)
    gaussian_means = anchor_xyz_full[:, :, None, :] + displacements  # (B, K+N, 4, 3)

    # Strip the first K=4 gripper anchors so output covers the scene PCD only —
    # mirrors the SMITH script's strip-last behavior, adjusted for gripper-first concat.
    weights = weights[:, K:]
    gaussian_means = gaussian_means[:, K:, :, :]
    anchor_points = anchor_xyz_full[:, K:, :]

    probabilities = torch.nn.functional.softmax(weights, dim=1)
    if args.argmax_weight:
        sampled_idx = torch.argmax(probabilities, dim=1)             # (B,)
    else:
        sampled_idx = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    batch_idx = torch.arange(B, device=outputs.device)
    prediction = gaussian_means[batch_idx, sampled_idx]              # (B, 4, 3)

    return prediction, probabilities, anchor_points, gaussian_means


def downsample(pcd, num_points):
    """Random downsample to exactly num_points; pad with repeats if too few."""
    if pcd.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    replace = pcd.shape[0] < num_points
    idx = np.random.choice(pcd.shape[0], num_points, replace=replace)
    return pcd[idx]


# --- generic verbatim-copy helpers (used by the --rl_bench extras path) ---
def _strip_leading_singleton(arr):
    """Drop a leading batch dim of size 1 (the per-timestep npz convention)."""
    arr = np.asarray(arr)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _to_str(x):
    """Coerce a numpy/bytes/object scalar to a Python str for vlen-string storage."""
    if isinstance(x, np.ndarray):
        x = x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def _write_buf(f, name, buf):
    """Stack a per-timestep buffer and write it, handling string/object dtypes.

    Numeric arrays are written as-is (dtype preserved). String/object arrays
    (e.g. lang_goal) are stored as variable-length UTF-8 — h5py cannot create a
    dataset directly from a numpy object/str array otherwise.
    """
    arr = np.stack(buf, axis=0)
    if arr.dtype.kind in ("U", "S", "O"):
        flat = np.array([_to_str(x) for x in arr.reshape(-1)], dtype=object)
        f.create_dataset(
            name,
            data=flat.reshape(arr.shape),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        f.create_dataset(name, data=arr)


def visualize_with_viser_interactive(
    all_timestep_data,
    show_all_gmm_goals=False,
    show_gmm_modes=False,
    gmm_weight_threshold: float = 0.01,
    gmm_alpha_gamma: float = 0.5,
):
    """Interactive viser viewer with a timestep slider — matches the SMITH script."""
    import time

    import viser

    server = viser.ViserServer()
    print(
        "[viser] Open http://localhost:8080. Use the Timestep slider to scrub predictions. Ctrl+C to exit."
    )

    T = len(all_timestep_data)
    timestep_slider = server.gui.add_slider(
        "Timestep", min=0, max=T - 1, step=1, initial_value=0
    )
    mode_info = server.gui.add_markdown("**modes:** —") if show_gmm_modes else None

    def render_timestep(t):
        data = all_timestep_data[t]
        merged_pcd = data["merged_pcd"]
        has_gmm = "weights" in data

        server.scene.add_point_cloud(
            name="scene_pcd",
            points=merged_pcd,
            colors=np.tile([180, 180, 180], (merged_pcd.shape[0], 1)).astype(np.uint8),
            point_size=0.004,
        )

        if has_gmm:
            weights_np = data["weights"].cpu().numpy().flatten()
            w_norm = (weights_np - weights_np.min()) / (
                weights_np.max() - weights_np.min() + 1e-8
            )
            anchor_colors = np.zeros((len(w_norm), 3), dtype=np.uint8)
            anchor_colors[:, 0] = (w_norm * 255).astype(np.uint8)
            anchor_colors[:, 2] = ((1 - w_norm) * 255).astype(np.uint8)
            server.scene.add_point_cloud(
                name="gmm_anchors",
                points=data["anchor_points"].cpu().numpy().reshape(-1, 3),
                colors=anchor_colors,
                point_size=0.008,
            )

            if show_all_gmm_goals:
                all_goals = data["gmm_all_components"].squeeze(0).cpu().numpy()
                N = all_goals.shape[0]
                # Softmax weights are highly peaked — most anchors carry near-zero
                # mass. Filter to anchors >= gmm_weight_threshold * peak so the
                # rendered cloud is actually meaningful, then fade-by-weight
                # within the kept set using the configured gamma.
                max_w = weights_np.max() + 1e-12
                keep = weights_np >= gmm_weight_threshold * max_w  # (N,) bool
                if keep.any():
                    kept_goals = all_goals[keep].reshape(-1, 3)
                    kept_alpha_anchor = np.power(weights_np[keep] / max_w, gmm_alpha_gamma)
                    kept_alpha = np.repeat(kept_alpha_anchor.astype(np.float32), 4)[:, None]
                    bright_green = np.array([0, 200, 0], dtype=np.float32)
                    bg_gray = np.array([180, 180, 180], dtype=np.float32)
                    goal_colors = (kept_alpha * bright_green + (1.0 - kept_alpha) * bg_gray).astype(np.uint8)
                    server.scene.add_point_cloud(
                        name="all_gmm_goals",
                        points=kept_goals,
                        colors=goal_colors,
                        point_size=0.014,
                    )

            server.scene.add_point_cloud(
                name="predicted_goal",
                points=data["prediction"].cpu().numpy().reshape(-1, 3),
                colors=np.tile([144, 238, 144], (4, 1)).astype(np.uint8),
                point_size=0.018,
            )

        server.scene.add_point_cloud(
            name="ground_truth_goal_gripper",
            points=data["gt_goal"].reshape(-1, 3),
            colors=np.tile([0, 0, 0], (4, 1)).astype(np.uint8),
            point_size=0.022,
        )

        # Current gripper (red), 4-pt — same source the network sees as input.
        if "present_gripper" in data:
            server.scene.add_point_cloud(
                name="present_gripper",
                points=np.asarray(data["present_gripper"]).reshape(-1, 3),
                colors=np.tile([255, 0, 0], (4, 1)).astype(np.uint8),
                point_size=0.022,
            )

        # Halo-collapsed GMM modes: each mode's 4 keypoints in a distinct color
        # (orange/cyan/yellow/magenta by mode index), big points so they stand
        # out over the green all-goals cloud. One cloud, re-written each frame —
        # there is always >=1 mode, so no stale modes linger.
        if show_gmm_modes and "gmm_modes" in data:
            modes = np.asarray(data["gmm_modes"])          # (K, 4, 3)
            mode_w = np.asarray(data["gmm_mode_weights"])  # (K,)
            palette = np.array(
                [[255, 140, 0], [0, 200, 255], [255, 255, 0], [255, 0, 255]],
                dtype=np.uint8,
            )
            pts, cols = [], []
            for k in range(modes.shape[0]):
                if mode_w[k] <= 0:
                    continue
                pts.append(modes[k].reshape(-1, 3))
                cols.append(np.tile(palette[k % len(palette)], (4, 1)))
            if pts:
                server.scene.add_point_cloud(
                    name="gmm_modes",
                    points=np.concatenate(pts, 0).astype(np.float32),
                    colors=np.concatenate(cols, 0).astype(np.uint8),
                    point_size=0.03,
                )
            if mode_info is not None:
                active = [f"{w:.3f}" for w in mode_w if w > 0]
                mode_info.content = f"**modes ({len(active)}):** " + ", ".join(active)

    @timestep_slider.on_update
    def _(_):
        render_timestep(timestep_slider.value)

    render_timestep(0)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[viser] Exiting.")
        server.stop()


def reduce_gmm_to_modes(all_goals, all_weights, radius=0.03, max_modes=3, w_thresh=1e-4):
    """Collapse the per-anchor GMM (N components) into a few spatial modes.

    The high-level GMM puts weight on every scene anchor, so a single physical
    goal is smeared across many neighbouring anchors (a "halo"): the top
    candidates by weight are typically <1 cm apart and describe ONE goal, while
    at a genuine transition the anchors split into two clumps 8-38 cm apart.

    This recovers the distinct modes via weight-ordered greedy NMS in
    goal-centroid space: the highest-weight not-yet-used anchor seeds a mode and
    absorbs every not-yet-used anchor whose goal centroid lies within `radius`
    metres of it; repeat until all anchors are claimed.

      - mode weight = SUM of member anchor weights (total probability mass of the
        mode — this un-does the halo's weight-splitting, unlike the per-anchor
        weight, so it is a meaningful per-mode confidence and the modes' weights
        still ~sum to 1).
      - mode goal   = weight-weighted average of member anchor goals (4, 3)
        (the consensus goal of the cluster).

    Args:
        all_goals:   (T, N, 4, 3) per-anchor goal keypoints for one demo.
        all_weights: (T, N)       per-anchor softmax weights (sum ~ 1 over N).
        radius:      merge distance in metres (anchors within this collapse into
                     one mode).
        max_modes:   K — output is sorted by mode weight descending and
                     zero-padded / truncated to this many modes (empty slots
                     have weight 0). Truncation drops the lowest-mass modes
                     without renormalising the survivors.
        w_thresh:    ignore anchors with weight below this (almost all are 0).

    Returns:
        modes:        (T, max_modes, 4, 3) float32  (zero-padded)
        mode_weights: (T, max_modes)       float32  (0.0 = empty slot)
    """
    T, N = all_weights.shape
    modes = np.zeros((T, max_modes, 4, 3), dtype=np.float32)
    mode_weights = np.zeros((T, max_modes), dtype=np.float32)

    for t in range(T):
        w = all_weights[t]
        keep = np.where(w > w_thresh)[0]
        if keep.size == 0:
            continue
        g = all_goals[t, keep]                 # (n, 4, 3)
        wk = w[keep].astype(np.float64)        # (n,)
        cent = g.mean(axis=1)                  # (n, 3) goal centroids

        used = np.zeros(keep.size, dtype=bool)
        collected = []                         # list of (mode_weight, mode_goal(4,3))
        for idx in np.argsort(wk)[::-1]:       # seeds in descending weight
            if used[idx]:
                continue
            d = np.linalg.norm(cent - cent[idx], axis=1)
            members = np.where((d <= radius) & (~used))[0]
            used[members] = True
            mw = wk[members].sum()
            mg = (g[members] * wk[members][:, None, None]).sum(axis=0) / mw
            collected.append((mw, mg.astype(np.float32)))

        collected.sort(key=lambda x: x[0], reverse=True)
        for k, (mw, mg) in enumerate(collected[:max_modes]):
            modes[t, k] = mg
            mode_weights[t, k] = mw

    return modes, mode_weights


def snap_nearest_mode_to_gt(modes, mode_weights, gt_goals):
    """Replace, per frame, the active mode whose goal centroid is nearest to the
    GT goal with the EXACT GT goal (`goal_gripper_pts`).

    Rationale: the realized mode (the one the demo actually moved toward) is the
    only mode the low-level action loss supervises, so giving it a clean GT
    target — instead of the high-level's noisy prediction — yields a sharper
    goal-follower. Cruise frames (1 active mode) -> that mode becomes GT;
    transition frames (>=2 active modes) -> the realized mode becomes GT while
    the alternative mode(s) stay as the high-level prediction (multimodality
    preserved). Mode weights are untouched (only positions move).

    Args:
        modes:        (T, K, 4, 3) float32  — from reduce_gmm_to_modes
        mode_weights: (T, K)       float32  — 0 marks an empty/padded slot
        gt_goals:     (T, 4, 3)    float32  — the GT goal_gripper_pts per frame
    Returns:
        modes with the nearest active mode of each frame overwritten by GT.
    """
    modes = modes.copy()
    T = modes.shape[0]
    gt_cent = gt_goals.mean(axis=1)          # (T, 3)
    mode_cent = modes.mean(axis=2)           # (T, K, 3)
    for t in range(T):
        active = np.where(mode_weights[t] > 0)[0]
        if active.size == 0:
            continue
        d = np.linalg.norm(mode_cent[t, active] - gt_cent[t], axis=1)
        k_star = active[int(np.argmin(d))]
        modes[t, k_star] = gt_goals[t].astype(modes.dtype)
    return modes


def process_demo_dir(demo_dir, network, text_embed, args):
    """Process a directory of per-timestep .npz files and write a consolidated .h5.

    Phase 1 — sequential CPU load of all T npz files into preallocated arrays.
    Phase 2 — chunked batched forward pass through ArticubotNetwork.
    Phase 3 — optional viser viz built from the populated CPU arrays.
    Phase 4 — h5 write.
    """
    npz_files = sorted(
        [f for f in os.listdir(demo_dir) if f.endswith(".npz")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    T_total = len(npz_files)
    T = T_total if args.max_timesteps is None else min(T_total, args.max_timesteps)
    npz_files = npz_files[:T]

    N_ANCHORS = args.num_points
    K = 4  # gripper points

    scene_pcds = np.zeros((T, N_ANCHORS, 3), dtype=np.float32)
    gripper_pcds = np.zeros((T, K, 3), dtype=np.float32)
    gt_goals = np.zeros((T, K, 3), dtype=np.float32)

    gmm_pred_goals = np.zeros((T, K, 3), dtype=np.float32)
    gmm_all_goals = np.zeros((T, N_ANCHORS, K, 3), dtype=np.float32)
    gmm_all_weights = np.zeros((T, N_ANCHORS), dtype=np.float32)

    # Camera mapping.
    #   mimicgen (default) : 2 cams (agentview, wrist) — RGB stored (H,W,3) uint8.
    #   --rl_bench         : 4 cams (front, L-shoulder, R-shoulder, wrist) —
    #                        keys use {cam}_rgb/{cam}_depth/{cam}_camera_extrinsics
    #                        /{cam}_camera_intrinsics; RGB is (3,H,W) float.
    # Each spec = (name, depth_key, rgb_key, extr_key, intr_key, rgb_is_chw_float).
    if args.rl_bench:
        cam_specs = [
            ("front",          "front_depth",          "front_rgb",
             "front_camera_extrinsics",          "front_camera_intrinsics",          True),
            ("left_shoulder",  "left_shoulder_depth",  "left_shoulder_rgb",
             "left_shoulder_camera_extrinsics",  "left_shoulder_camera_intrinsics",  True),
            ("right_shoulder", "right_shoulder_depth", "right_shoulder_rgb",
             "right_shoulder_camera_extrinsics", "right_shoulder_camera_intrinsics", True),
            ("wrist",          "wrist_depth",          "wrist_rgb",
             "wrist_camera_extrinsics",          "wrist_camera_intrinsics",          True),
        ]
    else:
        cam_specs = [
            ("agentview", "depth_agentview", "rgb_agentview",
             "agentview_extrinsics", "agentview_intrinsics", False),
            ("wrist",     "depth_wrist",     "rgb_wrist",
             "wrist_extrinsics",     "wrist_intrinsics",     False),
        ]

    obs_bufs = {
        "goal_gripper_pts": [],
        "point_cloud": [],
        "present_gripper_pts": [],
        "state": [],
    }
    for i in range(len(cam_specs)):
        for suffix in ("depth", "extrinsic", "image", "intrinsic"):
            obs_bufs[f"cam{i}_{suffix}"] = []
    act_delta = []

    # --rl_bench: copy every remaining npz key verbatim into obs/<key>.
    # `consumed_keys` are the ones already handled above (cameras + the standard
    # obs fields + action); everything else (eef_pos, eef_quat, gripper_qpos,
    # lang_goal, episode_idx, reward, terminal, timeout, per-cam point clouds...)
    # is copied as-is.
    if args.rl_bench:
        generic_bufs = defaultdict(list)
        consumed_keys = {"action", "point_cloud", "state", "goal_gripper_pcd", "gripper_pcd"}
        for _, k_depth, k_rgb, k_extr, k_intr, _ in cam_specs:
            consumed_keys.update((k_depth, k_rgb, k_extr, k_intr))

    def _depth_to_mm(d):
        d = np.squeeze(d).astype(np.float32)
        return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

    def _to_hwc_uint8(rgb):
        """Coerce RGB to (H,W,3) uint8. Handles (3,H,W) float (RL Bench)
        and (H,W,3) uint8 (mimicgen) inputs."""
        a = np.asarray(rgb)
        if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
            a = np.transpose(a, (1, 2, 0))
        return np.clip(a, 0, 255).astype(np.uint8)

    # ---- Phase 1: load all timesteps from disk into CPU arrays ----
    for t, fname in enumerate(npz_files):
        # allow_pickle=True is required because some RL Bench npz keys (e.g.
        # lang_goal) are stored as object/pickled arrays; the generic --rl_bench
        # copy loop touches every key, so they must be unpicklable.
        data = np.load(os.path.join(demo_dir, fname), allow_pickle=True)
        merged_pcd = data["point_cloud"][0]       # (N, 3)
        # Fallback: if `gripper_pcd` is absent (e.g. RL Bench-converted dataset),
        # derive the 4-pt current gripper from `state` via the RVT template.
        if "gripper_pcd" in data.files:
            gripper_pcd = data["gripper_pcd"][0]      # (4, 3)
        else:
            s = data["state"][0]                       # (8,) [xyz, qx,qy,qz,qw, open]
            gripper_pcd = build_gripper_pcd_from_pose_np(
                s[:7][None].astype(np.float32),
                np.array([[float(s[7])]], np.float32),
                _GRIPPER_TEMPLATE_4,
            )[0]                                        # (4, 3)
        gt_goal = data["goal_gripper_pcd"][0]     # (4, 3)

        if merged_pcd.shape[0] != args.num_points:
            merged_pcd = downsample(merged_pcd, args.num_points)

        scene_pcds[t] = merged_pcd.astype(np.float32)
        gripper_pcds[t] = gripper_pcd.astype(np.float32)
        gt_goals[t] = gt_goal.astype(np.float32)

        for i, (_, k_depth, k_rgb, k_extr, k_intr, rgb_chw) in enumerate(cam_specs):
            obs_bufs[f"cam{i}_depth"].append(_depth_to_mm(data[k_depth][0]))
            obs_bufs[f"cam{i}_extrinsic"].append(data[k_extr][0].astype(np.float32))
            obs_bufs[f"cam{i}_image"].append(
                _to_hwc_uint8(data[k_rgb][0]) if rgb_chw else data[k_rgb][0]
            )
            obs_bufs[f"cam{i}_intrinsic"].append(data[k_intr][0].astype(np.float32))
        obs_bufs["goal_gripper_pts"].append(gt_goal.astype(np.float32))
        obs_bufs["point_cloud"].append(scene_pcds[t])
        obs_bufs["present_gripper_pts"].append(gripper_pcds[t])
        obs_bufs["state"].append(data["state"][0].astype(np.float32))
        act_delta.append(data["action"][0].astype(np.float64))

        if args.rl_bench:
            for key in data.files:
                if key in consumed_keys:
                    continue
                generic_bufs[key].append(_strip_leading_singleton(data[key]))

    # ---- Phase 2: batched forward passes through the network ----
    bs = args.batch_size
    for start in range(0, T, bs):
        end = min(start + bs, T)
        b = end - start

        scene_b = torch.from_numpy(scene_pcds[start:end]).to("cuda", non_blocking=True)
        gripper_b = torch.from_numpy(gripper_pcds[start:end]).to("cuda", non_blocking=True)
        text_b = text_embed.expand(b, -1).contiguous()

        prediction, weights, _anchor_points, gmm_components = infer_gmm(
            network, scene_b, gripper_b, text_b, args
        )

        gmm_pred_goals[start:end] = prediction.cpu().numpy()
        gmm_all_goals[start:end] = gmm_components.cpu().numpy()
        gmm_all_weights[start:end] = weights.cpu().numpy()

    # ---- Phase 3 (optional): viser visualization ----
    if args.visualize or args.visualize_all_gmm_goals or args.visualize_gmm_modes:
        # When showing modes, reduce the per-anchor GMM with the SAME radius /
        # max_modes the dataset writer uses, so the viewer shows exactly what
        # gets stored as obs/gmm_modes.
        if args.visualize_gmm_modes:
            viz_modes, viz_mode_w = reduce_gmm_to_modes(
                gmm_all_goals, gmm_all_weights,
                radius=args.mode_radius, max_modes=args.max_modes,
            )
        all_timestep_data = []
        for t in range(T):
            entry = {
                "merged_pcd": scene_pcds[t],
                # anchor_points == scene_pcds (gripper anchors are stripped in infer_gmm)
                "anchor_points": torch.from_numpy(scene_pcds[t]).unsqueeze(0),
                "weights": torch.from_numpy(gmm_all_weights[t]).unsqueeze(0),
                "prediction": torch.from_numpy(gmm_pred_goals[t]).unsqueeze(0),
                "gmm_all_components": torch.from_numpy(gmm_all_goals[t]).unsqueeze(0),
                "gt_goal": gt_goals[t],
                "present_gripper": gripper_pcds[t],   # (K=4, 3) current gripper
            }
            if args.visualize_gmm_modes:
                entry["gmm_modes"] = viz_modes[t]            # (K, 4, 3)
                entry["gmm_mode_weights"] = viz_mode_w[t]    # (K,)
            all_timestep_data.append(entry)
        if len(all_timestep_data) > 0:
            visualize_with_viser_interactive(
                all_timestep_data,
                show_all_gmm_goals=args.visualize_all_gmm_goals,
                show_gmm_modes=args.visualize_gmm_modes,
                gmm_weight_threshold=args.gmm_weight_threshold,
                gmm_alpha_gamma=args.gmm_alpha_gamma,
            )

    # ---- Phase 4: write the consolidated h5 ----
    demo_name = os.path.basename(demo_dir)
    if getattr(args, "gmm_output_dir", None):
        out_h5 = os.path.join(args.gmm_output_dir, demo_name + ".h5")
    else:
        out_h5 = os.path.join(os.path.dirname(demo_dir), demo_name + ".h5")
    act_delta_arr = np.stack(act_delta, axis=0)

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("action/delta", data=act_delta_arr)
        f.create_dataset("action/hybrid", data=act_delta_arr.astype(np.float32))
        for key, buf in sorted(obs_bufs.items()):
            f.create_dataset(f"obs/{key}", data=np.stack(buf, axis=0))
        f.create_dataset("obs/gmm_pred_goal", data=gmm_pred_goals)
        f.create_dataset("obs/gmm_all_goals", data=gmm_all_goals)
        f.create_dataset("obs/gmm_all_weights", data=gmm_all_weights)

        # Optional: reduce the per-anchor GMM to a few spatial modes and store
        # them alongside the full distribution (gmm_all_* are kept regardless).
        if getattr(args, "save_modes_separately", False):
            gmm_modes, gmm_mode_weights = reduce_gmm_to_modes(
                gmm_all_goals,
                gmm_all_weights,
                radius=args.mode_radius,
                max_modes=args.max_modes,
            )
            # Optional: snap each frame's realized mode (nearest active mode to
            # the GT goal) onto the exact GT goal, for a clean low-level target.
            if getattr(args, "snap_nearest_mode_to_gt", False):
                gmm_modes = snap_nearest_mode_to_gt(
                    gmm_modes, gmm_mode_weights, gt_goals
                )
            f.create_dataset("obs/gmm_modes", data=gmm_modes)
            f.create_dataset("obs/gmm_mode_weights", data=gmm_mode_weights)
        for i in range(len(cam_specs)):
            f.create_dataset(f"_physical/cam{i}_extrinsic", data=obs_bufs[f"cam{i}_extrinsic"][0])
            f.create_dataset(f"_physical/cam{i}_intrinsic", data=obs_bufs[f"cam{i}_intrinsic"][0])

        if args.rl_bench:
            # Raw absolute control, verbatim under its original name (in addition
            # to action/delta + action/hybrid, which hold the same values).
            f.create_dataset("obs/action", data=act_delta_arr.astype(np.float32))
            # Every remaining original npz key, verbatim (strings -> vlen utf-8).
            for key in sorted(generic_bufs):
                _write_buf(f, f"obs/{key}", generic_bufs[key])

    return gmm_pred_goals, gmm_all_goals, gmm_all_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root containing demo_*/ subdirectories of .npz files",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to a Lightning .ckpt for the trained Articubot model",
    )
    parser.add_argument(
        "--text_embed_cache",
        type=str,
        default=None,
        help="Optional .npy with a (1152,) SigLIP text embedding, applied to the "
        "whole dataset. If unset, zeros are used (matches use_text_embed: False at "
        "training time). Ignored when --use_lang_goal is set.",
    )
    parser.add_argument(
        "--use_lang_goal",
        action="store_true",
        default=False,
        help="Condition the high-level model on each demo's own `lang_goal` "
        "string: embed the distinct captions fresh with SigLIP (canonical "
        "max_length padding) and feed the matching per-demo embedding to the "
        "forward pass. Use ONLY when --ckpt_path was trained with "
        "dataset.add_language_cond=True. Independent of --rl_bench; takes "
        "precedence over --text_embed_cache.",
    )
    parser.add_argument("--num_points", type=int, default=4500)
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        help="Network input channels. 4 = xyz + gripper-mask (use_rgb=False).",
    )
    parser.add_argument("--use_rgb", action="store_true", default=False)
    parser.add_argument(
        "--rl_bench", action="store_true", default=False,
        help="Treat the dataset as the RL Bench converted format: 4 cameras "
             "(front, left_shoulder, right_shoulder, wrist) mapped to the GROOT "
             "cam0..cam3 schema (RGB -> (H,W,3) uint8, depth -> uint16 mm). Also "
             "copies every remaining npz key verbatim into obs/<key> and stores "
             "the raw action at obs/action. Default uses the 2-cam mimicgen "
             "layout (agentview + wrist) and the original fixed schema."
    )
    parser.add_argument(
        "--argmax_weight",
        type=int,
        default=1,
        help="1 = pick the highest-weight anchor; 0 = sample anchor multinomially.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Timesteps batched per forward pass. Larger = faster, until GPU OOM.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch viser viewer (localhost:8080) showing anchors + sampled goal.",
    )
    parser.add_argument(
        "--visualize_all_gmm_goals",
        action="store_true",
        help="Also render all N GMM component goals colored by weight.",
    )
    parser.add_argument(
        "--visualize_gmm_modes",
        action="store_true",
        help="Also render the halo-collapsed GMM modes (from reduce_gmm_to_modes, "
        "using --mode_radius / --max_modes) — each mode's 4 keypoints in a "
        "distinct color, with per-mode weights shown in the GUI. Lets you see "
        "1 mode on cruise frames and 2-3 at transitions.",
    )
    parser.add_argument(
        "--gmm_weight_threshold",
        type=float,
        default=0.01,
        help="Min weight (as fraction of peak) for a GMM component goal to be drawn. Lower => more points shown.",
    )
    parser.add_argument(
        "--gmm_alpha_gamma",
        type=float,
        default=0.5,
        help="Gamma exponent on (weight/max_weight) when shading kept goals. Lower => brighter mid-range.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="After --start_demo filtering, only process this many demo_* directories.",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=None,
        help="Only process this many timesteps per demo.",
    )
    parser.add_argument(
        "--start_demo",
        type=int,
        default=0,
        help="Skip demo_* directories with index < start_demo (0-based). "
        "Use this to resume after the original B=1 job has produced some outputs.",
    )
    parser.add_argument(
        "--gmm_output_dir",
        type=str,
        default=None,
        help="Optional directory for the consolidated h5 outputs. If not set, "
        "h5 files are written next to the source demo_N/ directories (in-place). "
        "When set, the directory is created if missing and h5s land there instead.",
    )
    parser.add_argument(
        "--save_modes_separately",
        action="store_true",
        default=False,
        help="Also reduce the per-anchor GMM to a few spatial modes and write "
        "obs/gmm_modes (T, max_modes, 4, 3) + obs/gmm_mode_weights "
        "(T, max_modes) into each h5, alongside (not replacing) gmm_all_goals/"
        "gmm_all_weights. Mode weight = SUM of member anchor weights; mode goal "
        "= weight-weighted average of member goals.",
    )
    parser.add_argument(
        "--mode_radius",
        type=float,
        default=0.03,
        help="Merge radius in metres for GMM mode reduction: anchors whose goal "
        "centroids are within this distance collapse into one mode. Only used "
        "with --save_modes_separately.",
    )
    parser.add_argument(
        "--max_modes",
        type=int,
        default=3,
        help="Max modes (K) stored per timestep; output is sorted by mode weight "
        "descending and zero-padded, smallest modes dropped beyond K. Only used "
        "with --save_modes_separately.",
    )
    parser.add_argument(
        "--snap_nearest_mode_to_gt",
        action="store_true",
        default=False,
        help="After reducing to modes, overwrite (per frame) the active mode "
        "nearest the GT goal (goal_gripper_pts) with the EXACT GT goal — a clean "
        "target for the realized mode, while alternative transition modes stay "
        "as the high-level prediction. Requires --save_modes_separately. Default "
        "off (modes are the raw high-level prediction).",
    )
    args = parser.parse_args()

    model_cfg = build_model_cfg(in_channels=args.in_channels, use_rgb=args.use_rgb)
    network = load_articubot(args.ckpt_path, model_cfg)

    # Text conditioning for the high-level model. Three mutually exclusive modes:
    #   --use_lang_goal    : per-demo embedding computed fresh from each demo's
    #                        lang_goal (matches dataset.add_language_cond training).
    #   --text_embed_cache : one fixed embedding for the whole dataset.
    #   neither            : zeros (matches use_text_embed: False training).
    lang_goal_text_embed = None  # callable(demo_dir) -> (1, 1152) cuda tensor
    text_embed = None
    if args.use_lang_goal:
        from transformers import AutoModel, AutoProcessor

        from lfd3d.datasets.rgb_text_feature_gen import get_siglip_text_embedding

        print("Per-demo language conditioning ON: loading SigLIP for lang_goal...")
        _siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        _processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        _lang_cache = {}  # caption -> (1, 1152) cuda tensor (SigLIP runs once per caption)

        def lang_goal_text_embed(demo_dir):
            caption = read_demo_lang_goal(demo_dir)
            if caption not in _lang_cache:
                emb = get_siglip_text_embedding(
                    caption, siglip=_siglip, siglip_processor=_processor, device="cpu"
                ).astype(np.float32)  # (1152,)
                _lang_cache[caption] = (
                    torch.from_numpy(emb).float().unsqueeze(0).to("cuda")
                )
                print(f"[lang_goal] embedded caption: {caption!r}")
            return _lang_cache[caption]
    elif args.text_embed_cache and os.path.exists(args.text_embed_cache):
        text_embed_np = np.load(args.text_embed_cache).astype(np.float32)
        print(f"Loaded text embedding from {args.text_embed_cache}")
        text_embed = torch.from_numpy(text_embed_np).float().unsqueeze(0).to("cuda")
    else:
        text_embed_np = np.zeros(1152, dtype=np.float32)
        print("Using zero text embedding (use_text_embed: False at training time).")
        text_embed = torch.from_numpy(text_embed_np).float().unsqueeze(0).to("cuda")

    entries = os.listdir(args.dataset_dir)
    demo_dirs = sorted(
        [
            e
            for e in entries
            if e.startswith("demo_")
            and os.path.isdir(os.path.join(args.dataset_dir, e))
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    if args.start_demo > 0:
        demo_dirs = [d for d in demo_dirs if int(d.split("_")[1]) >= args.start_demo]
    if args.max_files is not None:
        demo_dirs = demo_dirs[: args.max_files]

    if args.gmm_output_dir:
        os.makedirs(args.gmm_output_dir, exist_ok=True)

    print(
        f"Processing {len(demo_dirs)} demo directories from {args.dataset_dir} "
        f"(start_demo={args.start_demo}, batch_size={args.batch_size}, rl_bench={args.rl_bench})"
    )
    for demo_name in tqdm(demo_dirs):
        demo_path = os.path.join(args.dataset_dir, demo_name)
        if args.gmm_output_dir:
            out_h5_check = os.path.join(args.gmm_output_dir, demo_name + ".h5")
        else:
            out_h5_check = os.path.join(args.dataset_dir, demo_name + ".h5")
        # Resume guard for dataset gen: skip demos already written. But in any
        # visualization mode the viewer only launches *inside* process_demo_dir,
        # so skipping a cached h5 would silently never show anything — always
        # re-process when visualizing.
        viz_mode = args.visualize or args.visualize_all_gmm_goals or args.visualize_gmm_modes
        if os.path.exists(out_h5_check) and not viz_mode:
            print(f"  {demo_name}: skipping (already exists at {out_h5_check})")
            continue
        demo_text_embed = (
            lang_goal_text_embed(demo_path) if args.use_lang_goal else text_embed
        )
        goals, all_goals, all_weights = process_demo_dir(
            demo_path, network, demo_text_embed, args
        )
        print(
            f"  {demo_name}: gmm_pred_goal {goals.shape}, "
            f"gmm_all_goals {all_goals.shape}, gmm_all_weights {all_weights.shape}"
        )
