#!/usr/bin/env python3
"""
Per-rate command scale for policies trained on achieved-delta labels.

Models trained on the legacy rate-ablation arms learned to predict the ACHIEVED
per-step eef delta (what the controller did), not the COMMAND that produced it.
At inference the eval script treats the prediction as a command, so the arm
under-moves by the controller's tracking ratio. The corrected arms (controller
inversion) hold, for the same steps, the command that actually reproduces each
achieved delta -- so the multiplier the eval must apply to a predicted achieved
delta to turn it into a command is the regression of command on achieved delta:

    cmd ~= K * achieved        K = sum(a.c) / sum(a.a)      (fit per rate, pos / rot separately)

fitted over unsaturated steps with non-trivial motion. Also reported: the forward
gain g = sum(a.c)/sum(c.c) (achieved per unit command), the correlation, and
K restricted to the 20 Hz native source where the two must agree with the
measured OSC tracking ratio (~0.23 pos / ~0.18 rot).

    python measure_cmd_scale.py --task hammer_cleanup_d1 [--legacy_dir ... --invert_dir ...]
"""
import argparse
import json
import os

import h5py
import numpy as np


def load(path):
    out = {}
    with h5py.File(path, "r") as h:
        rate = json.loads(h["data"].attrs["env_args"])["env_kwargs"]["control_freq"]
        for ep in h["data"]:
            g = h["data"][ep]
            out[ep] = dict(actions=g["actions"][()],
                           sat=g["invert_sat"][()] if "invert_sat" in g else None)
    return rate, out


def fit(a, c):
    """K (cmd per achieved), g (achieved per cmd), corr, over flattened axes."""
    a = a.ravel(); c = c.ravel()
    K = float(np.dot(a, c) / max(np.dot(a, a), 1e-12))
    g = float(np.dot(a, c) / max(np.dot(c, c), 1e-12))
    r = float(np.corrcoef(a, c)[0, 1]) if a.std() > 0 and c.std() > 0 else float("nan")
    return K, g, r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--rates", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    p.add_argument("--legacy_dir", default="/data/theya/data/rate_ablation/arms_achieved_legacy")
    p.add_argument("--invert_dir", default="/data/theya/data/rate_ablation/arms")
    p.add_argument("--min_motion", type=float, default=0.02,
                   help="ignore steps whose achieved |pos| or |rot| (normalised) is below this")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    print("{:>5s} {:>7s}  {:>8s} {:>8s} {:>6s}  |  {:>8s} {:>8s} {:>6s}   {:>6s} {:>6s}".format(
        "rate", "steps", "K_pos", "g_pos", "r_pos", "K_rot", "g_rot", "r_rot", "sat%", "demos"))
    table = {}
    for rate in args.rates:
        lp = os.path.join(args.legacy_dir, "{}_{}hz.hdf5".format(args.task, rate))
        ip = os.path.join(args.invert_dir, "{}_{}hz.hdf5".format(args.task, rate))
        if not (os.path.exists(lp) and os.path.exists(ip)):
            print("{:>5d}  (missing {})".format(rate, "legacy" if not os.path.exists(lp) else "inverted"))
            continue
        _, leg = load(lp)
        _, inv = load(ip)
        with h5py.File(ip, "r") as h:
            if h["data"].attrs.get("action_mode", "achieved") != "invert":
                print("{:>5d}  (inverted file is not action_mode=invert yet, skipping)".format(rate)); continue
        A_pos, C_pos, A_rot, C_rot, nsat, ntot = [], [], [], [], 0, 0
        for ep in sorted(set(leg) & set(inv), key=lambda s: int(s[5:])):
            a = leg[ep]["actions"][:-1]; c = inv[ep]["actions"][:-1]
            n = min(len(a), len(c)); a, c = a[:n], c[:n]
            sat = inv[ep]["sat"][:n] if inv[ep]["sat"] is not None else np.zeros(n, bool)
            keep_pos = (~sat) & (np.linalg.norm(a[:, :3], axis=1) > args.min_motion)
            keep_rot = (~sat) & (np.linalg.norm(a[:, 3:6], axis=1) > args.min_motion)
            A_pos.append(a[keep_pos, :3]); C_pos.append(c[keep_pos, :3])
            A_rot.append(a[keep_rot, 3:6]); C_rot.append(c[keep_rot, 3:6])
            nsat += int(sat.sum()); ntot += n
        A_pos, C_pos, A_rot, C_rot = map(np.concatenate, (A_pos, C_pos, A_rot, C_rot))
        Kp, gp, rp = fit(A_pos, C_pos); Kr, gr, rr = fit(A_rot, C_rot)
        print("{:>5d} {:>7d}  {:>8.3f} {:>8.3f} {:>6.3f}  |  {:>8.3f} {:>8.3f} {:>6.3f}   {:>5.1f}% {:>6d}".format(
            rate, ntot, Kp, gp, rp, Kr, gr, rr, 100.0 * nsat / max(ntot, 1), len(set(leg) & set(inv))))
        table[str(rate)] = dict(K_pos=Kp, K_rot=Kr, g_pos=gp, g_rot=gr, r_pos=rp, r_rot=rr,
                                steps=int(ntot), saturated_frac=nsat / max(ntot, 1))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(task=args.task, table=table), f, indent=1)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
