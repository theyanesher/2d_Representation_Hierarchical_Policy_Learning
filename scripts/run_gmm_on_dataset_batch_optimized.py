"""
Batch-optimized variant of run_gmm_on_dataset.py.

Same outputs as run_gmm_on_dataset.py — same H5 schema (obs/gmm_pred_goal,
obs/gmm_all_goals, obs/gmm_all_weights, all original obs/* fields,
action/delta, action/hybrid, _physical/cam*), same model loading, same
infer_gmm math (gripper-first concat + mask channel; strip-K=4 gripper
anchors; softmax weights; argmax or multinomial sampling).

Only difference: per-demo timesteps are batched through ArticubotNetwork in
chunks of --batch_size instead of one-at-a-time. The PointNet++ FPS Python
loop launches a fixed number of CUDA kernels regardless of batch size, so
per-sample cost drops sharply once B>1.

Phase 1: load all T npz files for the demo into preallocated CPU arrays.
Phase 2: run the network in chunks of --batch_size.
Phase 3: (optional) viser viz, fed from CPU arrays.
Phase 4: write the same H5 file as the original.

Resume support: --start_demo skips earlier demo_* directories (useful when
the original B=1 job has already produced some outputs).

Example:
  PYTHONNOUSERSITE=1 pixi run python scripts/run_gmm_on_dataset_batch_optimized.py \\
      --dataset_dir /scratch/.../Coffee_Task \\
      --ckpt_path   /path/to/.ckpt \\
      --start_demo 38 --max_files 1000 --batch_size 64
"""

import argparse
import os

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


def visualize_with_viser_interactive(
    all_timestep_data,
    show_all_gmm_goals=False,
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


def process_demo_dir(demo_dir, network, text_embed, args):
    """Process a directory of per-timestep .npz files and write a consolidated .h5.

    Phase 1 — sequential CPU load of all T npz files into preallocated arrays.
    Phase 2 — chunked batched forward pass through ArticubotNetwork.
    Phase 3 — optional viser viz built from the populated CPU arrays.
    Phase 4 — h5 write (identical schema to run_gmm_on_dataset.py).
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

    obs_bufs = {
        "cam0_depth": [],
        "cam0_extrinsic": [],
        "cam0_image": [],
        "cam0_intrinsic": [],
        "cam1_depth": [],
        "cam1_extrinsic": [],
        "cam1_image": [],
        "cam1_intrinsic": [],
        "goal_gripper_pts": [],
        "point_cloud": [],
        "present_gripper_pts": [],
        "state": [],
    }
    act_delta = []

    def _depth_to_mm(d):
        d = np.squeeze(d).astype(np.float32)
        return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)

    # ---- Phase 1: load all timesteps from disk into CPU arrays ----
    for t, fname in enumerate(npz_files):
        data = np.load(os.path.join(demo_dir, fname))
        merged_pcd = data["point_cloud"][0]       # (N, 3)
        gripper_pcd = data["gripper_pcd"][0]      # (4, 3)
        gt_goal = data["goal_gripper_pcd"][0]     # (4, 3)

        if merged_pcd.shape[0] != args.num_points:
            merged_pcd = downsample(merged_pcd, args.num_points)

        scene_pcds[t] = merged_pcd.astype(np.float32)
        gripper_pcds[t] = gripper_pcd.astype(np.float32)
        gt_goals[t] = gt_goal.astype(np.float32)

        obs_bufs["cam0_depth"].append(_depth_to_mm(data["depth_agentview"][0]))
        obs_bufs["cam0_extrinsic"].append(data["agentview_extrinsics"][0].astype(np.float32))
        obs_bufs["cam0_image"].append(data["rgb_agentview"][0])
        obs_bufs["cam0_intrinsic"].append(data["agentview_intrinsics"][0].astype(np.float32))
        obs_bufs["cam1_depth"].append(_depth_to_mm(data["depth_wrist"][0]))
        obs_bufs["cam1_extrinsic"].append(data["wrist_extrinsics"][0].astype(np.float32))
        obs_bufs["cam1_image"].append(data["rgb_wrist"][0])
        obs_bufs["cam1_intrinsic"].append(data["wrist_intrinsics"][0].astype(np.float32))
        obs_bufs["goal_gripper_pts"].append(gt_goal.astype(np.float32))
        obs_bufs["point_cloud"].append(scene_pcds[t])
        obs_bufs["present_gripper_pts"].append(gripper_pcds[t])
        obs_bufs["state"].append(data["state"][0].astype(np.float32))
        act_delta.append(data["action"][0].astype(np.float64))

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
    if args.visualize or args.visualize_all_gmm_goals:
        all_timestep_data = []
        for t in range(T):
            all_timestep_data.append(
                {
                    "merged_pcd": scene_pcds[t],
                    # anchor_points == scene_pcds (gripper anchors are stripped in infer_gmm)
                    "anchor_points": torch.from_numpy(scene_pcds[t]).unsqueeze(0),
                    "weights": torch.from_numpy(gmm_all_weights[t]).unsqueeze(0),
                    "prediction": torch.from_numpy(gmm_pred_goals[t]).unsqueeze(0),
                    "gmm_all_components": torch.from_numpy(gmm_all_goals[t]).unsqueeze(0),
                    "gt_goal": gt_goals[t],
                }
            )
        if len(all_timestep_data) > 0:
            visualize_with_viser_interactive(
                all_timestep_data,
                show_all_gmm_goals=args.visualize_all_gmm_goals,
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
        f.create_dataset("_physical/cam0_extrinsic", data=obs_bufs["cam0_extrinsic"][0])
        f.create_dataset("_physical/cam0_intrinsic", data=obs_bufs["cam0_intrinsic"][0])
        f.create_dataset("_physical/cam1_extrinsic", data=obs_bufs["cam1_extrinsic"][0])
        f.create_dataset("_physical/cam1_intrinsic", data=obs_bufs["cam1_intrinsic"][0])

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
        help="Optional .npy with a (1152,) SigLIP text embedding. "
        "If unset, zeros are used (matches use_text_embed: False at training time).",
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
    args = parser.parse_args()

    model_cfg = build_model_cfg(in_channels=args.in_channels, use_rgb=args.use_rgb)
    network = load_articubot(args.ckpt_path, model_cfg)

    if args.text_embed_cache and os.path.exists(args.text_embed_cache):
        text_embed_np = np.load(args.text_embed_cache).astype(np.float32)
        print(f"Loaded text embedding from {args.text_embed_cache}")
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
        f"(start_demo={args.start_demo}, batch_size={args.batch_size})"
    )
    for demo_name in tqdm(demo_dirs):
        demo_path = os.path.join(args.dataset_dir, demo_name)
        if args.gmm_output_dir:
            out_h5_check = os.path.join(args.gmm_output_dir, demo_name + ".h5")
        else:
            out_h5_check = os.path.join(args.dataset_dir, demo_name + ".h5")
        if os.path.exists(out_h5_check):
            print(f"  {demo_name}: skipping (already exists at {out_h5_check})")
            continue
        goals, all_goals, all_weights = process_demo_dir(
            demo_path, network, text_embed, args
        )
        print(
            f"  {demo_name}: gmm_pred_goal {goals.shape}, "
            f"gmm_all_goals {all_goals.shape}, gmm_all_weights {all_weights.shape}"
        )
