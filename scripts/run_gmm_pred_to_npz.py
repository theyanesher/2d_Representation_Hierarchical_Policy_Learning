"""Run a trained high-level articubot checkpoint over a demo npz tree and save
its GMM predictions to per-frame npz files — no h5 files are read or written.

Companion to run_gmm_on_dataset_batch_optimized.py (whose model loading /
input prep / inference helpers are imported and reused verbatim, so predicted
values are bit-identical to what the h5 pipeline would write). Instead of
consolidating everything into the 230GB GROOT-style h5 dataset, this dumps
ONLY the high-level model's outputs into a parallel npz tree that mirrors the
source dataset layout:

    <output_dir>/
        _generation_meta.json          # ckpt path + args for provenance
        demo_0/
            0.npz                      # keys suffixed by --key_suffix:
            1.npz                      #   gmm_pred_goal_<sfx>   (1, 4, 3)
            ...                        #   gmm_all_goals_<sfx>   (1, N, 4, 3)
        demo_1/                        #   gmm_all_weights_<sfx> (1, N)
            ...

The leading singleton dim matches the per-frame npz convention of the source
dataset and the EXTRA_KEYPOINTS goal trees, so the same alignment/injection
tooling applies later if these predictions are ever merged into h5s.

Intended use: --ckpt_path points at an RDP-goal-trained high-level run and
--key_suffix rdp, producing e.g. EXTRA_KEYPOINTS/Coffee_D2_GMM_PRED/. How
these predictions feed the low-level (snap-to-GT etc.) is deliberately
deferred — this script only materializes the raw distribution.

Resume: a demo is skipped when its output dir already contains one npz per
source frame. --start_demo / --max_files work as in the h5 generator.

Example:
  PYTHONNOUSERSITE=1 pixi run python scripts/run_gmm_pred_to_npz.py \\
      --dataset_dir /local/.../Coffee_D2_npz \\
      --ckpt_path   /path/to/periodic-epoch=epoch=99.ckpt \\
      --output_dir  /local/.../Coffee_D2_GMM_PRED \\
      --key_suffix rdp --batch_size 164
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from run_gmm_on_dataset_batch_optimized import (
    build_model_cfg,
    downsample,
    infer_gmm,
    load_articubot,
)


def process_demo_to_npz(demo_dir, out_demo_dir, network, text_embed, args):
    """Load one demo's frames, run batched GMM inference, write per-frame npz."""
    npz_files = sorted(
        [f for f in os.listdir(demo_dir) if f.endswith(".npz")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    T_total = len(npz_files)
    T = T_total if args.max_timesteps is None else min(T_total, args.max_timesteps)
    npz_files = npz_files[:T]

    N = args.num_points
    scene_pcds = np.zeros((T, N, 3), dtype=np.float32)
    gripper_pcds = np.zeros((T, 4, 3), dtype=np.float32)

    # Phase 1: load inference inputs only (point cloud + current gripper).
    # The high-level model consumes NO images — cameras are never read here.
    for t, fname in enumerate(npz_files):
        data = np.load(os.path.join(demo_dir, fname), allow_pickle=True)
        merged_pcd = data["point_cloud"][0]
        if merged_pcd.shape[0] != N:
            merged_pcd = downsample(merged_pcd, N)
        scene_pcds[t] = merged_pcd.astype(np.float32)
        gripper_pcds[t] = data["gripper_pcd"][0].astype(np.float32)

    # Phase 2: batched forward passes (same infer_gmm as the h5 generator).
    pred_goals = np.zeros((T, 4, 3), dtype=np.float32)
    all_goals = np.zeros((T, N, 4, 3), dtype=np.float32)
    all_weights = np.zeros((T, N), dtype=np.float32)

    bs = args.batch_size
    for start in range(0, T, bs):
        end = min(start + bs, T)
        b = end - start
        scene_b = torch.from_numpy(scene_pcds[start:end]).to(args.device, non_blocking=True)
        gripper_b = torch.from_numpy(gripper_pcds[start:end]).to(args.device, non_blocking=True)
        text_b = text_embed.expand(b, -1).contiguous()

        prediction, weights, _anchors, gmm_components = infer_gmm(
            network, scene_b, gripper_b, text_b, args
        )
        pred_goals[start:end] = prediction.cpu().numpy()
        all_goals[start:end] = gmm_components.cpu().numpy()
        all_weights[start:end] = weights.cpu().numpy()

    # Phase 3: one npz per source frame, same stem, leading singleton dim.
    sfx = args.key_suffix
    os.makedirs(out_demo_dir, exist_ok=True)
    for t, fname in enumerate(npz_files):
        np.savez(
            os.path.join(out_demo_dir, fname),
            **{
                f"gmm_pred_goal_{sfx}": pred_goals[t][None],     # (1, 4, 3)
                f"gmm_all_goals_{sfx}": all_goals[t][None],      # (1, N, 4, 3)
                f"gmm_all_weights_{sfx}": all_weights[t][None],  # (1, N)
            },
        )
    return T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root containing demo_*/ subdirectories of .npz files")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Lightning .ckpt of the trained high-level articubot model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root for the prediction npz tree (demo_N/t.npz created inside)")
    parser.add_argument("--key_suffix", type=str, default="rdp",
                        help="Suffix naming the goal source the checkpoint was trained on "
                             "(keys become gmm_pred_goal_<sfx>, gmm_all_goals_<sfx>, ...)")
    parser.add_argument("--text_embed_cache", type=str, default=None,
                        help="Optional (1152,) .npy SigLIP embedding; zeros if unset "
                             "(matches use_text_embed: False training)")
    parser.add_argument("--num_points", type=int, default=4500)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--use_rgb", action="store_true", default=False)
    parser.add_argument("--argmax_weight", type=int, default=1,
                        help="1 = argmax anchor for gmm_pred_goal; 0 = multinomial sample")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--start_demo", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda (default) or cpu (slow; smoke tests only)")
    args = parser.parse_args()

    model_cfg = build_model_cfg(in_channels=args.in_channels, use_rgb=args.use_rgb)
    network = load_articubot(args.ckpt_path, model_cfg, device=args.device)

    if args.text_embed_cache and os.path.exists(args.text_embed_cache):
        text_embed_np = np.load(args.text_embed_cache).astype(np.float32)
        print(f"Loaded text embedding from {args.text_embed_cache}")
    else:
        text_embed_np = np.zeros(1152, dtype=np.float32)
        print("Using zero text embedding (use_text_embed: False at training time).")
    text_embed = torch.from_numpy(text_embed_np).float().unsqueeze(0).to(args.device)

    demo_dirs = sorted(
        [e for e in os.listdir(args.dataset_dir)
         if e.startswith("demo_") and os.path.isdir(os.path.join(args.dataset_dir, e))],
        key=lambda x: int(x.split("_")[1]),
    )
    if args.start_demo > 0:
        demo_dirs = [d for d in demo_dirs if int(d.split("_")[1]) >= args.start_demo]
    if args.max_files is not None:
        demo_dirs = demo_dirs[: args.max_files]

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "_generation_meta.json"), "w") as f:
        json.dump(
            {
                "ckpt_path": os.path.abspath(args.ckpt_path),
                "key_suffix": args.key_suffix,
                "argmax_weight": args.argmax_weight,
                "num_points": args.num_points,
                "dataset_dir": os.path.abspath(args.dataset_dir),
            },
            f,
            indent=2,
        )

    print(f"Processing {len(demo_dirs)} demos from {args.dataset_dir} "
          f"(key_suffix={args.key_suffix}, batch_size={args.batch_size}, device={args.device})")
    n_done, n_skipped = 0, 0
    for demo_name in tqdm(demo_dirs):
        demo_path = os.path.join(args.dataset_dir, demo_name)
        out_demo_dir = os.path.join(args.output_dir, demo_name)

        # Resume: skip demos whose output already has one npz per source frame.
        n_src = len([f for f in os.listdir(demo_path) if f.endswith(".npz")])
        if args.max_timesteps is not None:
            n_src = min(n_src, args.max_timesteps)
        if os.path.isdir(out_demo_dir):
            n_out = len([f for f in os.listdir(out_demo_dir) if f.endswith(".npz")])
            if n_out >= n_src:
                n_skipped += 1
                continue

        T = process_demo_to_npz(demo_path, out_demo_dir, network, text_embed, args)
        n_done += 1
        print(f"  {demo_name}: wrote {T} frames -> {out_demo_dir}")

    print(f"[done] {n_done} demo(s) generated, {n_skipped} already complete.")
