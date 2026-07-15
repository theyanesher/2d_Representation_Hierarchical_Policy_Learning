"""Additive GT / predicted goal-mixing variant of LazyArticuBotDataset.

This is a PURE SUBCLASS — it imports and inherits LazyArticuBotDataset and only
adds new behaviour through super() calls. It does NOT modify the base dataset,
any existing config, or any existing training script. Every existing run is
therefore completely unaffected; this feature is reachable only when a task
config sets `_target_` to this class and provides `gt_mix_p`.

Behaviour
---------
With per-training-item probability `gt_mix_p` (a stationary Bernoulli flipped
once per __getitem__), the predicted high-level GMM goal distribution
(`gmm_all_goals` / `gmm_all_weights`) is replaced by a sparse "ground-truth"
goal set built from the demo's own `obs/goal_gripper_pts`:

  * cruise frame      -> 1 mode  = present GT goal,           weight 1.0
  * transition frame  -> 2 modes = [present, neighbor] GT goal, triangular
                          weights w_neighbor = p_max * (1 - d/(radius+1)),
                          w_present = 1 - w_neighbor

where a "transition" is a frame within +/- `gt_transition_radius` of a
goal-change (keyframe boundary), `d` is the distance to the nearest such change,
and `neighbor` is the goal on the other side of it (the upcoming goal before the
change, the previous goal at/after it). This mirrors the high-level pipeline's
transition window + triangular weighting (npy_dataset._compute_swap_meta), but
expressed as explicit 2-mode weighted-cross-attention weights.

With probability (1 - gt_mix_p) the predicted GMM is passed through untouched.

The sparse GT goals occupy slots 0/1 of the existing (N, 4, 3) GMM container with
all other slots at weight 0 — so `gmm_top_k` and weighted-cross-attention work
unchanged (the GT goals carry all the weight and are always selected; empty
slots are log(1e-8)-masked). No policy / architecture / shape changes.

`obs/goal_gripper_pts` is read directly from the episode h5 files at init purely
to build these GT goals; it is NOT added to the model's shape_meta, so the policy
input is identical to the plain GMM-WCA run.
"""

import numpy as np
import torch
from pathlib import Path

from diffusion_policy.dataset.lazy_articubot_dataset import LazyArticuBotDataset


class LazyArticuBotGtMixDataset(LazyArticuBotDataset):

    def __init__(self,
                 *args,
                 gt_mix_p=None,
                 gt_transition_radius=5,
                 gt_transition_pmax=0.5,
                 gt_goal_npz_dir=None,
                 gt_goal_npz_key='goal_gripper_pcd_rdp',
                 **kwargs):
        # Build the full base dataset exactly as usual (these gt_* kwargs are
        # captured here and never forwarded, so the parent sees nothing new).
        super().__init__(*args, **kwargs)

        self.gt_mix_p = gt_mix_p
        self.gt_transition_radius = int(gt_transition_radius)
        self.gt_transition_pmax = float(gt_transition_pmax)

        # Optional external GT goal source (RDP pipeline): when set, the GT
        # goals for the sparse mode set are read per-frame from
        # <gt_goal_npz_dir>/demo_N/<t>.npz under key gt_goal_npz_key (e.g. the
        # EXTRA_KEYPOINTS tree's goal_gripper_pcd_rdp) instead of the h5's
        # obs/goal_gripper_pts. Transition windows then follow that goal
        # schedule. The h5 files stay untouched either way.
        self.gt_goal_npz_dir = Path(gt_goal_npz_dir) if gt_goal_npz_dir else None
        self.gt_goal_npz_key = gt_goal_npz_key
        if self.gt_goal_npz_dir is not None and not self.gt_goal_npz_dir.is_dir():
            raise FileNotFoundError(
                f"gt_goal_npz_dir does not exist: {self.gt_goal_npz_dir}")

        # Locate the GMM goal/weight obs keys by their shape_meta type.
        self.gt_gmm_goals_key = None
        self.gt_gmm_weights_key = None
        for key, attr in self.shape_meta['obs'].items():
            if attr.get('type') == 'gmm_goals':
                self.gt_gmm_goals_key = key
            elif attr.get('type') == 'gmm_weights':
                self.gt_gmm_weights_key = key

        self._gt_mix_enabled = False
        if gt_mix_p is not None:
            if self.gt_gmm_goals_key is None or self.gt_gmm_weights_key is None:
                print(f"[LazyArticuBotGtMixDataset] gt_mix_p={gt_mix_p} requested but no "
                      f"gmm_goals/gmm_weights obs keys in shape_meta -> GT goal mixing DISABLED.")
            else:
                # Reconstruct the exact episode file list the base buffer used:
                # base builds it as sorted(glob("*.h5"))[:max_train_episodes], and
                # replay_buffer.n_episodes == that sliced length.
                h5_paths = sorted(Path(self.data_dir).glob("*.h5"))[: self.replay_buffer.n_episodes]
                self._precompute_gt_goal_mix(h5_paths)
                self._gt_mix_enabled = True
                gt_src = (f"npz:{self.gt_goal_npz_dir}[{self.gt_goal_npz_key}]"
                          if self.gt_goal_npz_dir is not None
                          else "h5:obs/goal_gripper_pts")
                print(f"[LazyArticuBotGtMixDataset] GT goal mixing ENABLED: p={gt_mix_p}, "
                      f"radius={self.gt_transition_radius}, p_max={self.gt_transition_pmax}, "
                      f"goals_key={self.gt_gmm_goals_key!r}, weights_key={self.gt_gmm_weights_key!r}, "
                      f"gt_source={gt_src}.")

    # ----------------------------------------------------------------------- #
    # Precompute, per GLOBAL frame, the sparse GT goal set (present + optional
    # neighbor) and its triangular mode weights. Reads obs/goal_gripper_pts
    # directly from each episode h5 (not via the model's shape_meta).
    # ----------------------------------------------------------------------- #
    def _precompute_gt_goal_mix(self, h5_paths):
        import h5py

        episode_ends = np.asarray(self.replay_buffer.episode_ends)
        total = int(episode_ends[-1]) if len(episode_ends) > 0 else 0

        goals_shape = self.shape_meta['obs'][self.gt_gmm_goals_key]['shape']  # [N, K, C]
        K = int(goals_shape[1])
        C = int(goals_shape[2])

        self._gt_present = np.zeros((total, K, C), dtype=np.float32)
        self._gt_neighbor = np.zeros((total, K, C), dtype=np.float32)
        self._gt_w_present = np.ones((total,), dtype=np.float32)   # default: cruise (weight 1)
        self._gt_w_neighbor = np.zeros((total,), dtype=np.float32)

        radius = self.gt_transition_radius
        pmax = self.gt_transition_pmax

        start = 0
        for ep_i, h5path in enumerate(h5_paths):
            end = int(episode_ends[ep_i])
            T = end - start
            if self.gt_goal_npz_dir is not None:
                # External GT source: one npz per frame, mirroring the dataset
                # layout (demo_N/<t>.npz), value shaped (1, K, C).
                stem = Path(h5path).stem
                goals = np.stack([
                    np.load(self.gt_goal_npz_dir / stem / f"{t}.npz")[self.gt_goal_npz_key][0]
                    for t in range(T)
                ]).astype(np.float32)  # (T, K, C)
            else:
                with h5py.File(h5path, 'r') as f:
                    if 'obs/goal_gripper_pts' not in f:
                        raise KeyError(
                            f"gt_mix requires obs/goal_gripper_pts but it is missing in {h5path}"
                        )
                    goals = f['obs/goal_gripper_pts'][:T].astype(np.float32)  # (T, K, C)

            self._gt_present[start:end] = goals

            # Keyframe boundaries = frames where the GT goal changes.
            transitions = [t for t in range(1, T)
                           if not np.allclose(goals[t], goals[t - 1], atol=1e-6)]
            if transitions:
                for t in range(T):
                    best_d, best_t = None, None
                    for tt in transitions:
                        d = abs(t - tt)
                        if d <= radius and (best_d is None or d < best_d):
                            best_d, best_t = d, tt
                    if best_t is None:
                        continue  # outside every window -> stays cruise (present @ 1.0)
                    # neighbor = goal on the other side of the nearest boundary.
                    # tt is the FIRST frame of the new goal: frames < tt carry the
                    # old goal (neighbor = upcoming), frames >= tt carry the new
                    # goal (neighbor = previous).
                    neighbor = goals[best_t] if t < best_t else goals[best_t - 1]
                    w_neighbor = pmax * (1.0 - best_d / (radius + 1))
                    gi = start + t
                    self._gt_neighbor[gi] = neighbor
                    self._gt_w_neighbor[gi] = w_neighbor
                    self._gt_w_present[gi] = 1.0 - w_neighbor
            start = end

        n_trans = int((self._gt_w_neighbor > 0).sum())
        print(f"[LazyArticuBotGtMixDataset] gt_mix precompute: {total} frames, "
              f"{n_trans} inside a +-{radius} transition window (2-goal); rest cruise (1-goal).")

    # ----------------------------------------------------------------------- #
    # Per-item Bernoulli(gt_mix_p): on heads, overwrite the predicted GMM goals/
    # weights for this window's obs frames with the sparse GT goal set.
    # ----------------------------------------------------------------------- #
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if getattr(self, "_gt_mix_enabled", False) and np.random.random() < self.gt_mix_p:
            self._apply_gt_goal_mix(idx, sample["obs"])
        return sample

    def _apply_gt_goal_mix(self, idx, obs):
        goals_key = self.gt_gmm_goals_key
        weights_key = self.gt_gmm_weights_key

        pred_goals = obs[goals_key]      # torch (To, N, K, C)
        pred_w = obs[weights_key]        # torch (To, N)
        To = int(pred_w.shape[0])

        # Map each obs position j -> global replay-buffer frame, mirroring how the
        # SequenceSampler lays real frames into the padded window
        # (positions [ssi, sei) <- buffer [bsi, bei); pad_before repeats the first).
        bsi, bei, ssi, sei = self.sampler.indices[idx]
        bsi = int(bsi); ssi = int(ssi); sei = int(sei)

        goals_out = torch.zeros_like(pred_goals)
        weights_out = torch.zeros_like(pred_w)
        for j in range(To):
            real_pos = min(max(j, ssi), sei - 1)
            gf = bsi + (real_pos - ssi)
            goals_out[j, 0] = torch.from_numpy(self._gt_present[gf].copy())
            weights_out[j, 0] = float(self._gt_w_present[gf])
            if self._gt_w_neighbor[gf] > 0.0:
                goals_out[j, 1] = torch.from_numpy(self._gt_neighbor[gf].copy())
                weights_out[j, 1] = float(self._gt_w_neighbor[gf])

        obs[goals_key] = goals_out
        obs[weights_key] = weights_out

    # GT goal mixing is a TRAIN-only augmentation; validation always uses the
    # predicted GMM goals (the deployment-time signal). The base implementation
    # returns a copy.copy of self (preserving this subclass), so we just flip the
    # flag off on the validation view.
    def get_validation_dataset(self):
        val_set = super().get_validation_dataset()
        val_set._gt_mix_enabled = False
        return val_set
