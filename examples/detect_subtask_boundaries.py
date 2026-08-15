#!/usr/bin/env python3
"""Detect boundary-only original frame indices from an episode NPZ or demo dir.

Examples:
    # Start scripts/serve_qwen_local.sh first; local Qwen is the default.
    pixi run python examples/detect_subtask_boundaries.py \
        episode.npz --rgb-key rgb

    OPENAI_API_KEY=... pixi run python examples/detect_subtask_boundaries.py \
        episode.npz --rgb-key rgb --provider openai

    GEMINI_API_KEY=... pixi run python examples/detect_subtask_boundaries.py \
        /data/theya/data/uncertainity_subgoal/D1/COFFEE_PREPERATION_D1/demo_0 \
        --rgb-key rgb_agentview --provider gemini
"""

import argparse
from pathlib import Path

import numpy as np

from subtask_boundaries import detect_subtask_boundaries, load_rgb_npz


def _numeric_npz_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.npz"))
    try:
        return sorted(files, key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(
            f"all per-step NPZ filenames must be integer frame indices: {directory}"
        ) from exc


def load_input(path: Path, rgb_key: str) -> np.ndarray:
    if path.is_file():
        return load_rgb_npz(path, rgb_key)
    if not path.is_dir():
        raise FileNotFoundError(path)
    step_files = _numeric_npz_files(path)
    if not step_files:
        raise ValueError(f"demo directory contains no NPZ step files: {path}")
    frames = []
    for step_path in step_files:
        with np.load(step_path, allow_pickle=False) as data:
            if rgb_key not in data:
                raise KeyError(f"{rgb_key!r} not found in {step_path}; available: {data.files}")
            frame = data[rgb_key]
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            frames.append(frame)
    return np.stack(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="episode NPZ or per-step demo directory")
    parser.add_argument("--rgb-key", default="rgb")
    parser.add_argument(
        "--provider",
        choices=["qwen", "qwen_cloud", "openai", "gemini"],
        default="qwen",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "default: qwen3.6-local for local Qwen, qwen3.6-flash for "
            "QwenCloud, gpt-5.4 for OpenAI, gemini-3.5-flash for Gemini"
        ),
    )
    parser.add_argument(
        "--qwen-base-url",
        default=None,
        help="local Qwen OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)",
    )
    parser.add_argument("--sample-every-n-frames", type=int, default=15)
    parser.add_argument("--refinement-radius", type=int, default=15)
    parser.add_argument("--refinement-stride", type=int, default=1)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument(
        "--stop-after-sparse-annotation",
        action="store_true",
        help="return coarse sampled-frame boundaries without dense refinement",
    )
    parser.add_argument("--min-boundary-distance-frames", type=int, default=0)
    parser.add_argument(
        "--sheet-overlap-frames",
        type=int,
        default=2,
        help="sampled frames repeated across consecutive sheets (default: 2)",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--logs-dir",
        default="logs/subtask_boundaries",
        help="directory for sparse VLM input/output logs",
    )
    args = parser.parse_args()

    frames = load_input(args.input, args.rgb_key)
    boundaries = detect_subtask_boundaries(
        frames,
        provider=args.provider,
        model=args.model,
        qwen_base_url=args.qwen_base_url,
        sample_every_n_frames=args.sample_every_n_frames,
        refine=not args.no_refine,
        stop_after_sparse_annotation=args.stop_after_sparse_annotation,
        refinement_radius=args.refinement_radius,
        refinement_stride=args.refinement_stride,
        min_boundary_distance_frames=args.min_boundary_distance_frames,
        sheet_overlap_frames=args.sheet_overlap_frames,
        instruction=args.instruction,
        logs_dir=args.logs_dir,
    )
    print(boundaries)


if __name__ == "__main__":
    main()
