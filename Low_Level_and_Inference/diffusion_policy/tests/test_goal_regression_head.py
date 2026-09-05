"""Tests for the Approach-2 regression aux head (``aux_head_type="regression"``).

Covers the parts that are easy to get subtly wrong:

  * ``goal_regression_target`` picking the right frame for each of the two
    ``aux_regression_frame`` values -- "absolute" and "relative_to_gripper";
  * ``GoalRegressionHead`` actually ignoring masked-out (invalid) tokens when
    mean-pooling, not just zero-weighting them in the sum;
  * ``goal_regression_loss`` behaving like plain MSE (zero iff pred==target);
  * the policy's ``_compute_goal_regression_loss`` wiring both frames through
    the SAME anchor stack ``_compute_goal_gmm_loss`` reads, and producing
    different numbers for the two frames on the same prediction (proof the
    flag actually changes what is being regressed against, not just a no-op).

Run directly (``pixi run python diffusion_policy/tests/test_goal_regression_head.py``)
or under pytest.
"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion_policy.model.flow_matching.goal_regression_head import (
    REGRESSION_FRAMES, GoalRegressionHead, goal_regression_loss,
    goal_regression_target,
)


# --------------------------------------------------------------------------- #
# goal_regression_target                                                       #
# --------------------------------------------------------------------------- #

def test_absolute_frame_targets_the_goal_directly():
    goal = torch.randn(2, 4, 3)
    present = torch.randn(2, 4, 3)
    assert torch.equal(goal_regression_target(goal, present, "absolute"), goal)


def test_relative_frame_targets_the_displacement_to_travel():
    goal = torch.randn(2, 4, 3)
    present = torch.randn(2, 4, 3)
    target = goal_regression_target(goal, present, "relative_to_gripper")
    assert torch.equal(target, goal - present)


def test_the_two_frames_disagree_whenever_the_gripper_has_moved():
    """Sanity check that "absolute" and "relative_to_gripper" are not
    accidentally the same computation."""
    goal = torch.randn(2, 4, 3)
    present = torch.randn(2, 4, 3)
    abs_t = goal_regression_target(goal, present, "absolute")
    rel_t = goal_regression_target(goal, present, "relative_to_gripper")
    assert not torch.allclose(abs_t, rel_t)
    # ... but they must agree when the gripper is already at the origin.
    zero_present = torch.zeros_like(present)
    assert torch.equal(goal_regression_target(goal, zero_present, "absolute"),
                       goal_regression_target(goal, zero_present, "relative_to_gripper"))


def test_unknown_frame_is_rejected():
    with pytest.raises(ValueError, match="unknown aux_regression_frame"):
        goal_regression_target(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), "bogus")


# --------------------------------------------------------------------------- #
# goal_regression_loss                                                         #
# --------------------------------------------------------------------------- #

def test_loss_is_zero_iff_prediction_matches_target():
    pred = torch.randn(5, 4, 3)
    assert float(goal_regression_loss(pred, pred)) == 0.0
    assert float(goal_regression_loss(pred, pred + 1.0)) > 0.0


def test_loss_matches_hand_computed_mse():
    pred = torch.randn(3, 4, 3)
    target = torch.randn(3, 4, 3)
    expected = ((pred - target) ** 2).sum(dim=-1).mean()
    assert torch.allclose(goal_regression_loss(pred, target), expected)


# --------------------------------------------------------------------------- #
# GoalRegressionHead                                                           #
# --------------------------------------------------------------------------- #

def test_head_output_shape():
    torch.manual_seed(0)
    head = GoalRegressionHead(token_dim=8, hidden_dim=16, n_keypoints=4)
    tokens = torch.randn(3, 6, 8)
    valid = torch.ones(3, 6, dtype=torch.bool)
    out = head(tokens, valid)
    assert tuple(out.shape) == (3, 4, 3)


def test_head_ignores_masked_out_tokens():
    """Corrupting tokens behind the mask must not move the pooled prediction --
    otherwise the head is leaking information through invalid anchors."""
    torch.manual_seed(0)
    head = GoalRegressionHead(token_dim=8, hidden_dim=16, n_keypoints=4)
    tokens = torch.randn(3, 6, 8)
    valid = torch.zeros(3, 6, dtype=torch.bool)
    valid[:, :3] = True

    baseline = head(tokens, valid)
    corrupted = tokens.clone()
    corrupted[:, 3:] = torch.randn(3, 3, 8) * 1000
    assert torch.allclose(baseline, head(corrupted, valid), atol=1e-5)


def test_head_handles_a_row_with_no_valid_tokens():
    """valid.sum(dim=1) can be 0 for a degenerate row; the clamp(min=1.0) in the
    pooling denominator must keep this finite, not NaN."""
    torch.manual_seed(0)
    head = GoalRegressionHead(token_dim=8, hidden_dim=16, n_keypoints=4)
    tokens = torch.randn(2, 6, 8)
    valid = torch.zeros(2, 6, dtype=torch.bool)
    valid[1] = True
    out = head(tokens, valid)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Policy wiring: both aux_regression_frame values through the real code path  #
# --------------------------------------------------------------------------- #

def _fake_tokens(B, To, K, n_cams, n_tok, D, seed):
    torch.manual_seed(seed)
    n_vis = To * n_cams * n_tok
    return dict(
        vis_tokens=torch.randn(B, n_vis, D),
        vis_xyz=torch.randn(B, n_vis, 3),
        vis_valid=torch.rand(B, n_vis) > 0.3,
        grip_tokens=torch.randn(B, To * K, D),
        grip_xyz=torch.randn(B, To * K, 3),
    )


def _stub_regression_policy(n_obs_steps, n_keypoints, frame, head):
    policy_mod = pytest.importorskip(
        "diffusion_policy.policy.flow_matching_dit_goal_gmm_policy",
        reason="needs the full policy stack (transformers/diffusers)")
    real = policy_mod.FlowMatchingDiTGoalGMMPolicy

    class Stub:
        _gmm_anchor_stack = real._gmm_anchor_stack
        _compute_goal_regression_loss = real._compute_goal_regression_loss

    stub = Stub()
    stub.n_obs_steps = n_obs_steps
    stub.n_keypoints = n_keypoints
    stub.aux_regression_frame = frame
    stub.regression_head = head
    return stub


def test_both_regression_frames_are_wired_and_give_different_losses():
    """Both values in REGRESSION_FRAMES must be usable end-to-end via
    ``_compute_goal_regression_loss``, and -- since the anchors (grip_xyz) are
    non-zero here -- the two frames must score the SAME prediction
    differently. If they always agreed, the frame flag would be a no-op."""
    B, To, K, n_cams, n_tok, D = 3, 2, 4, 2, 5, 8
    assert set(REGRESSION_FRAMES) == {"absolute", "relative_to_gripper"}

    head = GoalRegressionHead(token_dim=D, hidden_dim=16, n_keypoints=K)
    t = _fake_tokens(B, To, K, n_cams, n_tok, D, seed=0)
    goal = torch.randn(B, To, K, 3)

    losses = {}
    for frame in REGRESSION_FRAMES:
        policy = _stub_regression_policy(To, K, frame, head)
        losses[frame] = float(policy._compute_goal_regression_loss(
            t["vis_tokens"], t["vis_xyz"], t["vis_valid"],
            t["grip_tokens"], t["grip_xyz"], goal,
        ))
        assert torch.isfinite(torch.tensor(losses[frame]))

    assert losses["absolute"] != losses["relative_to_gripper"]


def test_regression_loss_matches_manual_computation_for_each_frame():
    """Cross-check ``_compute_goal_regression_loss`` against directly calling
    the same head + target + loss functions on the anchor-stacked tokens, so a
    reshape bug in the policy's stacking can't silently agree with itself."""
    B, To, K, n_cams, n_tok, D = 2, 2, 3, 2, 4, 8
    head = GoalRegressionHead(token_dim=D, hidden_dim=16, n_keypoints=K)
    t = _fake_tokens(B, To, K, n_cams, n_tok, D, seed=1)
    goal = torch.randn(B, To, K, 3)

    for frame in REGRESSION_FRAMES:
        policy = _stub_regression_policy(To, K, frame, head)
        tokens, _, valid = policy._gmm_anchor_stack(
            t["vis_tokens"], t["vis_xyz"], t["vis_valid"],
            t["grip_tokens"], t["grip_xyz"],
        )
        goal_t = goal[:, :To].reshape(B * To, K, 3)
        present_t = t["grip_xyz"].reshape(B * To, K, 3)

        pred = head(tokens, valid)
        target = goal_regression_target(goal_t, present_t, frame)
        expected = goal_regression_loss(pred, target)

        got = policy._compute_goal_regression_loss(
            t["vis_tokens"], t["vis_xyz"], t["vis_valid"],
            t["grip_tokens"], t["grip_xyz"], goal,
        )
        assert torch.allclose(got, expected, atol=1e-6), frame


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
