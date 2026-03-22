#!/usr/bin/env python3
"""
Compare rollout videos from two directories by stacking them vertically.

Output filenames encode the pass/fail result for (top, bottom):
  0_pass_pass.mp4, 1_pass_fail.mp4, 2_fail_pass.mp4, 3_fail_fail.mp4, ...

Usage:
  python scripts/compare_rollouts.py <dir1> <dir2> [--output media/compare] [--threshold 0.1]
"""

import argparse
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def find_rollout_videos(directory: Path) -> dict[int, tuple[Path, float]]:
    """Return {rollout_num: (path, score)} for all <num>_<score>.mp4 files."""
    videos = {}
    for f in directory.glob("*.mp4"):
        m = re.match(r"^(\d+)_(.+)\.mp4$", f.name)
        if m:
            rollout_num = int(m.group(1))
            try:
                score = float(m.group(2))
            except ValueError:
                print(f"  Warning: could not parse score from '{f.name}', skipping.")
                continue
            videos[rollout_num] = (f, score)
    return videos


def compare_label(score1: float, score2: float, threshold: float) -> str:
    """Return 'pass_pass', 'pass_fail', 'fail_pass', or 'fail_fail'."""
    top = "pass" if score1 >= threshold else "fail"
    bot = "pass" if score2 >= threshold else "fail"
    return f"{top}_{bot}"


def stack_videos(top: Path, bottom: Path, output: Path) -> bool:
    """Stack two videos vertically (top above bottom) using imageio."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        frames_top = iio.imread(str(top)) #, plugin="pyav")    # (T, H, W, C)
        frames_bot = iio.imread(str(bottom)) # , plugin="pyav")
        fps = 30 # iio.improps(str(top), plugin="pyav").fps or 30

        # Pad widths if they differ
        w = max(frames_top.shape[2], frames_bot.shape[2])
        def pad_width(frames, target_w):
            diff = target_w - frames.shape[2]
            if diff > 0:
                pad = np.zeros((frames.shape[0], frames.shape[1], diff, frames.shape[3]), dtype=frames.dtype)
                frames = np.concatenate([frames, pad], axis=2)
            return frames
        frames_top = pad_width(frames_top, w)
        frames_bot = pad_width(frames_bot, w)

        # Match lengths by repeating last frame of the shorter video
        t = max(len(frames_top), len(frames_bot))
        def extend(frames, length):
            if len(frames) < length:
                tail = np.repeat(frames[[-1]], length - len(frames), axis=0)
                frames = np.concatenate([frames, tail], axis=0)
            return frames
        frames_top = extend(frames_top, t)
        frames_bot = extend(frames_bot, t)

        stacked = np.concatenate([frames_top, frames_bot], axis=1)  # stack on H axis
        iio.imwrite(str(output), stacked, fps=fps, codec="h264")
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Create stacked comparison videos from two rollout directories.")
    parser.add_argument("dir1", type=Path, help="First rollout directory (top video)")
    parser.add_argument("dir2", type=Path, help="Second rollout directory (bottom video)")
    parser.add_argument("--output", type=Path, default=Path("media/compare"),
                        help="Output directory (default: media/compare)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for pass/fail (default: 0.5)")
    parser.add_argument("--rollouts", type=int, nargs="+",
                        help="Only process these rollout numbers (default: all common rollouts)")
    args = parser.parse_args()

    dir1, dir2 = args.dir1, args.dir2
    if not dir1.is_dir():
        sys.exit(f"Error: {dir1} is not a directory")
    if not dir2.is_dir():
        sys.exit(f"Error: {dir2} is not a directory")

    videos1 = find_rollout_videos(dir1)
    videos2 = find_rollout_videos(dir2)

    common = sorted(set(videos1) & set(videos2))
    if args.rollouts:
        common = sorted(set(common) & set(args.rollouts))

    if not common:
        sys.exit("No matching rollout videos found in both directories.")

    only_in_1 = sorted(set(videos1) - set(videos2))
    only_in_2 = sorted(set(videos2) - set(videos1))
    if only_in_1:
        print(f"Rollouts only in dir1 (skipped): {only_in_1}")
    if only_in_2:
        print(f"Rollouts only in dir2 (skipped): {only_in_2}")

    print(f"Processing {len(common)} rollouts -> {args.output}/")
    print(f"  dir1 (top):    {dir1}")
    print(f"  dir2 (bottom): {dir2}")
    print(f"  pass/fail threshold: {args.threshold}")

    success, failed = 0, []
    pass1, pass2 = 0, 0
    counts = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
    for i, num in enumerate(common):
        top_path, score1 = videos1[num]
        bottom_path, score2 = videos2[num]
        label = compare_label(score1, score2, args.threshold)
        output_path = args.output / f"{num}_{label}.mp4"
        print(f"  [{i+1}/{len(common)}] rollout {num}: score1={score1:.4g}  score2={score2:.4g}  -> {label}")
        if score1 >= args.threshold:
            pass1 += 1
        if score2 >= args.threshold:
            pass2 += 1
        counts[label] += 1
        if stack_videos(top_path, bottom_path, output_path):
            success += 1
        else:
            failed.append(num)

    n = len(common)
    print(f"\nDone: {success} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed rollouts: {failed}")
    print(f"\nPass rate:  dir1 (top):    {pass1}/{n}  |  dir2 (bottom): {pass2}/{n}")
    print(f"Breakdown:  pass_pass={counts['pass_pass']}  pass_fail={counts['pass_fail']}  fail_pass={counts['fail_pass']}  fail_fail={counts['fail_fail']}")


if __name__ == "__main__":
    main()
