#!/usr/bin/env python3
"""Find the best checkpoint for one or more training runs.

There is no automatic "best" checkpoint in these runs: the topk-by-score
checkpoint manager in train_diffusion_unet_hybrid_workspace.py is disabled
(env_runner/rollout is commented out, so test_mean_score is never computed).
Only epoch_N.ckpt (every training.checkpoint_every epochs) and latest.ckpt
are saved. "Best" here means: the epoch with the lowest val_loss in
logs.json.txt, mapped to the nearest saved checkpoint at or before that
epoch (since not every epoch has a checkpoint on disk).

Usage:
    python find_best_checkpoint.py outputs/2026.08.16/*_hammercleanup_D1_*
    python find_best_checkpoint.py outputs/2026.08.16/06.48.32_..._dinov2_DIT_..._goal_gmm_aux
    python find_best_checkpoint.py --metric train_action_mse_error outputs/2026.08.16/*
"""
import argparse
import glob
import json
import re
from pathlib import Path


def load_epoch_metric(run_dir: Path, metric: str):
    log_path = run_dir / "logs.json.txt"
    if not log_path.exists():
        return {}
    by_epoch = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric in row and "epoch" in row:
                # last value logged for a given epoch wins (val runs at
                # epoch end, so later entries are more complete)
                by_epoch[row["epoch"]] = row[metric]
    return by_epoch


def available_checkpoints(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return {}
    out = {}
    for p in ckpt_dir.glob("epoch_*.ckpt"):
        m = re.match(r"epoch_(\d+)\.ckpt$", p.name)
        if m and p.stat().st_size > 0:
            out[int(m.group(1))] = p
    return out


def closest_at_or_before(epoch: int, ckpts: dict):
    candidates = [e for e in ckpts if e <= epoch]
    if candidates:
        return max(candidates)
    if ckpts:
        return min(ckpts)  # nothing at/before it -> fall back to earliest saved
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="run output dirs (glob patterns ok)")
    ap.add_argument("--metric", default="val_loss",
                     help="metric key in logs.json.txt to minimize (default: val_loss)")
    ap.add_argument("--top", type=int, default=1,
                     help="show this many best epochs per run (default: 1)")
    args = ap.parse_args()

    dirs = []
    for pattern in args.run_dirs:
        matches = sorted(glob.glob(pattern))
        dirs.extend(Path(m) for m in matches) if matches else dirs.append(Path(pattern))

    for run_dir in dirs:
        if not run_dir.is_dir():
            print(f"[skip] not a directory: {run_dir}")
            continue
        by_epoch = load_epoch_metric(run_dir, args.metric)
        ckpts = available_checkpoints(run_dir)
        print(f"\n=== {run_dir.name} ===")
        if not by_epoch:
            print(f"  no '{args.metric}' found in logs.json.txt")
            continue
        if not ckpts:
            print("  no epoch_N.ckpt checkpoints on disk")
            continue

        ranked = sorted(by_epoch.items(), key=lambda kv: kv[1])
        for epoch, value in ranked[:args.top]:
            ckpt_epoch = closest_at_or_before(epoch, ckpts)
            flag = "" if ckpt_epoch == epoch else f"  (nearest saved: epoch {ckpt_epoch})"
            print(f"  epoch {epoch:>4}  {args.metric}={value:.4f}{flag}")
            print(f"    -> {ckpts[ckpt_epoch]}")


if __name__ == "__main__":
    main()
