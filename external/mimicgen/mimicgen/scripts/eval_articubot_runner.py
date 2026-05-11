#!/usr/bin/env python3
"""
Evaluate trained ArticuBot high-level and low-level policies using the same env runner
as train_articubot_workspace, with policy loading and inference via mimicgen.utils.control_robot.

- Creates env runner via Hydra (same as train_articubot_workspace eval).
- Loads low-level policy with control_robot.load_low_level_policy(exp_dir, checkpoint_name).
- Loads high-level policy with control_robot (load_multitask_high_level_model, or
  load_high_level_weighted_displacement_policy, or load_high_level_gmm_policy).
- Runner uses control_robot for inference when use_control_robot_inference=True (set in config).

Usage (from articubot-on-mimicgen repo root, with mimicgen on PYTHONPATH):
  python -m mimicgen.scripts.eval_articubot_runner \
    --config-path=equi_diffpo/config --config-name=eval_articubot_standalone \
    output_dir=./eval_out \
    dataset_path=/path/to/dataset.hdf5 \
    low_level_exp_dir=/path/to/low_level_exp \
    low_level_checkpoint=latest.ckpt \
    high_level_path=/path/to/high_level/model.pth \
    high_level=weighted_displacement
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure articubot-on-mimicgen (equi_diffpo, third_party) is on path when config_path is under it
def _setup_paths(config_path: str | None) -> None:
    if config_path is None:
        return
    p = Path(config_path).resolve()
    if not p.is_absolute():
        p = Path.cwd() / p
    # config_path is e.g. equi_diffpo/config -> parent is articubot-on-mimicgen root
    root = p.parent.parent
    if (root / "equi_diffpo").exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval ArticuBot policies with env runner + control_robot loading")
    parser.add_argument("--config-path", type=str, default="equi_diffpo/config",
                        help="Hydra config path (relative to cwd or absolute)")
    parser.add_argument("--config-name", type=str, default="eval_articubot_standalone",
                        help="Hydra config name")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. output_dir=./out high_level_path=/path")
    args = parser.parse_args()

    _setup_paths(args.config_path)

    import hydra
    from omegaconf import OmegaConf
    from termcolor import cprint

    from mimicgen.utils import control_robot

    # Resolvers used by task configs (e.g. get_max_steps)
    try:
        from eval import get_ws_x_center, get_ws_y_center
        max_steps = {
            "stack_d1": 400, "stack_three_d1": 400, "square_d0": 400, "square_d2": 400,
            "threading_d2": 400, "coffee_d2": 400, "three_piece_assembly_d2": 500,
            "hammer_cleanup_d1": 500, "mug_cleanup_d1": 500, "kitchen_d1": 800,
            "nut_assembly_d0": 500, "pick_place_d0": 1000, "coffee_preparation_d1": 800,
            "tool_hang": 700, "can": 400, "lift": 400, "square": 400,
        }
        OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps.get(x, 400), replace=True)
        OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
        OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
    except ImportError:
        OmegaConf.register_new_resolver("get_max_steps", lambda x: 400, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize(config_path=args.config_path, version_base=None):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)

    OmegaConf.resolve(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Create env runner (same as train_articubot_workspace eval)
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(output_dir),
        high_level=cfg.high_level,
        use_control_robot_inference=True,
        cat_idx=cfg.cat_idx,
    )

    # 2) Load low-level policy (control_robot style)
    if not cfg.low_level_exp_dir:
        cprint("low_level_exp_dir is required", "red")
        return 1
    cprint(f"Loading low-level policy from {cfg.low_level_exp_dir} / {cfg.low_level_checkpoint}", "cyan")
    low_level_policy = control_robot.load_low_level_policy(
        cfg.low_level_exp_dir,
        cfg.low_level_checkpoint,
    )
    low_level_policy.eval()
    low_level_policy.cuda()

    # 3) Load high-level policy (control_robot style)
    if not cfg.high_level_path:
        cprint("high_level_path is required", "red")
        return 1
    cprint(f"Loading high-level policy ({cfg.high_level}) from {cfg.high_level_path}", "cyan")
    if cfg.high_level == "multitask":
        high_level_policy, high_level_args = control_robot.load_multitask_high_level_model(cfg.high_level_path)
        env_runner.high_level_args = high_level_args
    elif cfg.high_level == "weighted_displacement":
        high_level_policy = control_robot.load_high_level_weighted_displacement_policy(
            cfg.high_level_path,
            use_color=cfg.get("use_pc_color", True),
            use_groupnorm=False,
        )
    elif cfg.high_level == "gmm":
        high_level_policy = control_robot.load_high_level_gmm_policy(cfg.high_level_path)
    else:
        cprint(f"Unknown high_level: {cfg.high_level}", "red")
        return 1
    high_level_policy.eval()
    high_level_policy.cuda()

    # 4) Run evaluation
    cprint("Running evaluation...", "yellow")
    runner_log = env_runner.run(high_level_policy, low_level_policy)

    cprint("---------------- Eval Results --------------", "magenta")
    for key, value in runner_log.items():
        if isinstance(value, float):
            cprint(f"  {key}: {value:.4f}", "magenta")
    return 0


if __name__ == "__main__":
    sys.exit(main())
