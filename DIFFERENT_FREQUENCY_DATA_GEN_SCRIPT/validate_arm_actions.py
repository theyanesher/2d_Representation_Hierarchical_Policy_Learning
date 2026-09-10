#!/usr/bin/env python3
"""
Validate the action labels of rate-ablation arm hdf5s.

Two checks, both independent of any policy:

  1. STATS   compare each arm's action magnitudes with the native 20 Hz source
             (same demos). For the 20 Hz arm, also a step-aligned comparison:
             arm step j <-> source step j, since base = 25*T_src + 1 ticks.
             Correctly labelled 20 Hz commands must look like the source ones
             (|pos| mean ~0.3, |rot| ~0.09, ~10-20% saturated), not ~4x smaller.

  2. REPLAY  (--replay) open-loop: reset to state_0 and step the arm's stored
             actions at the arm's control_freq. Report eef tracking error against
             the arm's own stored poses, and task success. A correct label set
             tracks to within a few mm; the legacy achieved-delta labels drift
             immediately (the arm moves ~1/4 of each commanded step at 20 Hz).

Usage
    python validate_arm_actions.py --task hammer_cleanup_d1                 # stats, all arms
    python validate_arm_actions.py --task hammer_cleanup_d1 --replay --n 30 # + replay
    python validate_arm_actions.py --task hammer_cleanup_d1 --arm_dir /data/theya/data/rate_ablation/arms_achieved_legacy --replay --n 30
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_SRC_DIR = "/data/theya/data/uncertainity_subgoal/D1/env_hdf5/core"
DEFAULT_ARM_DIR = "/data/theya/data/rate_ablation/arms"

_ENV = None


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    pos = np.linalg.norm(a[:, :3], axis=1)
    rot = np.linalg.norm(a[:, 3:6], axis=1)
    sat = np.mean(np.any(np.abs(a[:, :6]) >= 1.0 - 1e-6, axis=1))
    return dict(pos_mean=np.abs(a[:, :3]).mean(), pos_p95=np.percentile(pos, 95),
                rot_mean=np.abs(a[:, 3:6]).mean(), rot_p95=np.percentile(rot, 95), sat=sat)


def stats_report(src_path, arm_paths, n_demos):
    with h5py.File(src_path, "r") as hs:
        src_freq = json.loads(hs["data"].attrs["env_args"])["env_kwargs"]["control_freq"]
        src_acts = {}
        for arm_path in arm_paths:
            with h5py.File(arm_path, "r") as ha:
                for ep in list(ha["data"].keys())[:n_demos]:
                    if ep not in src_acts and ep in hs["data"]:
                        src_acts[ep] = hs["data"][ep]["actions"][()]
    ref = _stats(np.concatenate(list(src_acts.values())))
    print("\n== action magnitude (normalised OSC units) ==")
    print("{:<28s}{:>9s}{:>9s}{:>9s}{:>9s}{:>9s}{:>10s}".format(
        "dataset", "|pos|mean", "pos p95", "|rot|mean", "rot p95", "sat%", "mode"))
    print("{:<28s}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}{:>8.1f}%{:>10s}".format(
        "native source {} Hz".format(src_freq), ref["pos_mean"], ref["pos_p95"],
        ref["rot_mean"], ref["rot_p95"], 100 * ref["sat"], "-"))
    for arm_path in arm_paths:
        with h5py.File(arm_path, "r") as ha:
            rate = json.loads(ha["data"].attrs["env_args"])["env_kwargs"]["control_freq"]
            mode = ha["data"].attrs.get("action_mode", "achieved(legacy)")
            eps = list(ha["data"].keys())[:n_demos]
            acts = np.concatenate([ha["data"][ep]["actions"][()] for ep in eps])
            s = _stats(acts)
            print("{:<28s}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.3f}{:>8.1f}%{:>10s}".format(
                "arm {} Hz".format(rate), s["pos_mean"], s["pos_p95"], s["rot_mean"],
                s["rot_p95"], 100 * s["sat"], str(mode)))
            for key in ("invert_err_pos_median_mm", "invert_err_pos_p95_mm",
                        "invert_err_rot_p95_deg", "invert_saturated_frac"):
                if key in ha["data"].attrs:
                    print("    {:<26s}{:.3f}".format(key, float(ha["data"].attrs[key])))

            if rate == src_freq:
                # step-aligned: arm row j is the command issued at source step j
                d_pos, d_rot, cor = [], [], []
                for ep in eps:
                    a = ha["data"][ep]["actions"][()]
                    s_ = src_acts.get(ep)
                    if s_ is None:
                        continue
                    T_ = min(len(s_), len(a) - 1)
                    d_pos.append(np.abs(a[:T_, :3] - s_[:T_, :3]).ravel())
                    d_rot.append(np.abs(a[:T_, 3:6] - s_[:T_, 3:6]).ravel())
                    for ax in range(6):
                        x, y = a[:T_, ax], s_[:T_, ax]
                        if x.std() > 1e-6 and y.std() > 1e-6:
                            cor.append(np.corrcoef(x, y)[0, 1])
                d_pos = np.concatenate(d_pos); d_rot = np.concatenate(d_rot)
                print("    step-aligned vs source: |dpos| median {:.3f} p95 {:.3f}   "
                      "|drot| median {:.3f} p95 {:.3f}   per-axis corr median {:.3f}"
                      .format(np.median(d_pos), np.percentile(d_pos, 95),
                              np.median(d_rot), np.percentile(d_rot, 95), np.median(cor)))


# ---- replay ---------------------------------------------------------------- #
def _init_replay(env_meta, rate):
    global _ENV
    from resimulate_at_base_rate import _build_env
    _ENV = _build_env(env_meta, rate)


def _replay_demo(task):
    ep, model_xml, state0, actions, eef_pos = task
    try:
        from resimulate_at_base_rate import _reset_to
        env = _ENV
        _reset_to(env, model_xml, state0)
        errs = []
        success = False
        for t in range(len(actions) - 1):
            post, _, _, _ = env.step(np.asarray(actions[t], dtype=np.float64))
            errs.append(np.linalg.norm(np.asarray(post["robot0_eef_pos"]) - eef_pos[t + 1]))
            if not success and env._check_success():
                success = True
        errs = np.asarray(errs)
        return ep, dict(success=success, err_median=float(np.median(errs)),
                        err_p95=float(np.percentile(errs, 95)), err_final=float(errs[-1])), None
    except Exception as exc:
        return ep, None, "{}: {}".format(type(exc).__name__, exc)


def replay_report(arm_paths, n_demos, pool_size):
    print("\n== open-loop replay of stored actions at each arm's control_freq ==")
    print("{:<12s}{:>8s}{:>14s}{:>14s}{:>14s}{:>12s}".format(
        "arm", "demos", "eef err med", "eef err p95", "eef err final", "success"))
    ctx = mp.get_context("spawn")
    for arm_path in arm_paths:
        with h5py.File(arm_path, "r") as ha:
            meta = json.loads(ha["data"].attrs["env_args"])
            rate = meta["env_kwargs"]["control_freq"]
            eps = list(ha["data"].keys())[:n_demos]
            tasks = [(ep, ha["data"][ep].attrs["model_file"], ha["data"][ep]["states"][0],
                      ha["data"][ep]["actions"][()], ha["data"][ep]["obs/robot0_eef_pos"][()])
                     for ep in eps]
        t0 = time.time()
        res, errs = [], []
        with ctx.Pool(min(pool_size, len(tasks)), initializer=_init_replay,
                      initargs=(meta, rate)) as pool:
            for ep, r, err in pool.imap_unordered(_replay_demo, tasks):
                (errs if err else res).append((ep, err) if err else r)
        if errs:
            print("  {} Hz: {} replay errors, e.g. {}".format(rate, len(errs), errs[:2]))
        if res:
            print("{:<12s}{:>8d}{:>11.1f} mm{:>11.1f} mm{:>11.1f} mm{:>11.0%}   ({:.0f}s)".format(
                "{} Hz".format(rate), len(res),
                1e3 * np.median([r["err_median"] for r in res]),
                1e3 * np.median([r["err_p95"] for r in res]),
                1e3 * np.median([r["err_final"] for r in res]),
                np.mean([r["success"] for r in res]), time.time() - t0), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--rates", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    p.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    p.add_argument("--arm_dir", default=DEFAULT_ARM_DIR)
    p.add_argument("--n", type=int, default=100, help="demos per arm to use")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--pool_size", type=int, default=16)
    args = p.parse_args()

    src = os.path.join(args.src_dir, args.task + ".hdf5")
    arms = [os.path.join(args.arm_dir, "{}_{}hz.hdf5".format(args.task, r)) for r in args.rates]
    arms = [a for a in arms if os.path.exists(a)] or sys.exit("no arm files under {}".format(args.arm_dir))
    print("source: {}\narms  : {}".format(src, args.arm_dir))
    stats_report(src, arms, args.n)
    if args.replay:
        replay_report(arms, args.n, args.pool_size)


if __name__ == "__main__":
    main()
