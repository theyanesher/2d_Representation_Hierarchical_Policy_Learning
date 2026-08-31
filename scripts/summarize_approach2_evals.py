#!/usr/bin/env python3
"""Combine per-seed Approach 2 results into JSON and readable text summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ORDINAL_SEEDS = {"1ST": 100000, "2ND": 150000, "3RD": 250000}


def infer_seed(result_dir: Path) -> int | None:
    name = result_dir.name.upper()
    numeric = re.search(r"(?:^|_)(\d+)_SEED(?:$|_)", name)
    if numeric:
        return int(numeric.group(1))
    for ordinal, seed in ORDINAL_SEEDS.items():
        if re.search(rf"(?:^|_){ordinal}_SEED(?:$|_)", name):
            return seed

    args_path = result_dir / "args.json"
    if args_path.is_file():
        try:
            value = json.loads(args_path.read_text()).get("seed")
            return int(value) if value is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return None


def load_result_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows, 1
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            invalid += 1
    return rows, invalid


def summarize_seed(seed: int, result_dir: Path) -> dict[str, Any]:
    rows, invalid_rows = load_result_rows(result_dir / "results.jsonl")
    successes = sum(bool(row.get("success", False)) for row in rows)
    rewards = [
        float(row["reward"])
        for row in rows
        if isinstance(row.get("reward"), (int, float))
        and math.isfinite(float(row["reward"]))
    ]
    episodes = len(rows)
    return {
        "seed": seed,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes else None,
        "mean_reward": statistics.fmean(rewards) if rewards else None,
        "invalid_result_rows": invalid_rows,
        "result_dir": str(result_dir),
    }


def discover_seed_summaries(eval_dir: Path) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, dict[str, Any]] = {}
    for results_path in sorted(eval_dir.rglob("results.jsonl")):
        result_dir = results_path.parent
        if any("_RESUME_" in part.upper() for part in result_dir.parts):
            continue
        seed = infer_seed(result_dir)
        if seed is None:
            continue
        candidate = summarize_seed(seed, result_dir)
        previous = by_seed.get(seed)
        if previous is None or candidate["episodes"] > previous["episodes"]:
            by_seed[seed] = candidate
    return by_seed


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def build_summary(
    eval_dir: Path, checkpoint: str, expected_seeds: list[int]
) -> dict[str, Any]:
    found = discover_seed_summaries(eval_dir)
    seeds = [found[seed] for seed in expected_seeds if seed in found]
    extra_seeds = [found[seed] for seed in sorted(found) if seed not in expected_seeds]
    seeds.extend(extra_seeds)

    total_episodes = sum(row["episodes"] for row in seeds)
    total_successes = sum(row["successes"] for row in seeds)
    seed_rates = finite_values(seeds, "success_rate")
    mean_rewards = finite_values(seeds, "mean_reward")
    missing = [seed for seed in expected_seeds if seed not in found]

    return {
        "checkpoint": checkpoint,
        "eval_dir": str(eval_dir),
        "expected_seeds": expected_seeds,
        "complete": not missing,
        "missing_seeds": missing,
        "seed_summaries": seeds,
        "combined": {
            "num_seeds": len(seeds),
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "pooled_success_rate": (
                total_successes / total_episodes if total_episodes else None
            ),
            "mean_seed_success_rate": (
                statistics.fmean(seed_rates) if seed_rates else None
            ),
            "std_seed_success_rate": (
                statistics.pstdev(seed_rates) if len(seed_rates) > 1 else 0.0
                if seed_rates
                else None
            ),
            "mean_seed_reward": (
                statistics.fmean(mean_rewards) if mean_rewards else None
            ),
        },
    }


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_text(summary: dict[str, Any]) -> str:
    lines = [
        f"Checkpoint: {summary['checkpoint']}",
        f"Complete:   {'yes' if summary['complete'] else 'no'}",
        "",
        "Seed      Episodes  Successes  Success rate  Mean reward",
        "--------- --------- ----------  ------------  -----------",
    ]
    for row in summary["seed_summaries"]:
        reward = row["mean_reward"]
        reward_text = "n/a" if reward is None else f"{reward:.6f}"
        lines.append(
            f"{row['seed']:<9} {row['episodes']:>9} {row['successes']:>10}  "
            f"{format_percent(row['success_rate']):>12}  {reward_text:>11}"
        )
    combined = summary["combined"]
    lines.extend(
        [
            "",
            f"Total: {combined['total_successes']}/{combined['total_episodes']} successes",
            f"Pooled success rate:    {format_percent(combined['pooled_success_rate'])}",
            f"Mean seed success rate: {format_percent(combined['mean_seed_success_rate'])}",
            f"Std seed success rate:  {format_percent(combined['std_seed_success_rate'])}",
        ]
    )
    if summary["missing_seeds"]:
        lines.append(
            "Missing seeds: " + ", ".join(str(seed) for seed in summary["missing_seeds"])
        )
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-seeds", type=int, nargs="+", default=[100000, 150000, 250000]
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()

    eval_dir = args.eval_dir.resolve()
    summary = build_summary(eval_dir, args.checkpoint, args.expected_seeds)
    json_output = args.json_output or eval_dir / "summary_all_seeds.json"
    text_output = args.text_output or eval_dir / "summary_all_seeds.txt"
    atomic_write(json_output, json.dumps(summary, indent=2) + "\n")
    atomic_write(text_output, render_text(summary))

    print(render_text(summary), end="")
    print(f"Combined JSON: {json_output}")
    print(f"Combined text: {text_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
