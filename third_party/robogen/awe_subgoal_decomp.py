"""
AWE (Automatic Waypoint Extraction) subgoal decomposition for trajectory
keypoint extraction (https://pypi.org/project/waypoint-extraction/).

Companion to subgoal_decomp.py (curvature/jerk), bayesian_subgoal_decomp.py
(BOCPD), and rdp_subgoal_decomp.py (RDP-family). Same public contract:
compute_awe_subgoal_gripper_pcd returns an expanded goal_gripper_pcd of shape
(T, 4, 3) plus the switch-index array, so the output is a drop-in
interchangeable replacement for the BOCPD/RDP `goal_gripper_pcd`.

Unlike RDP/curvature/BOCPD (proprioceptive-only, purely geometric heuristics),
AWE picks the sparsest set of frame indices ("waypoints") such that linearly
interpolating the end-effector pose between consecutive waypoints
reconstructs the full trajectory within `err_threshold`. Each frame's subgoal
is the gripper keypoints of the next waypoint it hasn't passed yet
(inclusive) -- goals repeat until the demo passes each waypoint, the same
convention subgoal_decomp.py/rdp_subgoal_decomp.py use for goal_gripper_pcd.

NOTE on the AWE package API: despite what you may have seen elsewhere, the
installed `waypoint_extraction` package does NOT expose an `extract_waypoints`
function. It exposes two selection algorithms:
    waypoint_extraction.dp_waypoint_selection(...)      # optimal, O(T^3) - slow
    waypoint_extraction.greedy_waypoint_selection(...)  # near-optimal, faster
Both work purely geometrically (env=None) as long as `actions[:, -1]` carries
the gripper open/close command and `gt_states` carries eef pos/quat -- no live
simulator is required. This module uses `greedy` by default; pass method="dp"
only for short demos (a few hundred frames at most).

Packaging quirk worked around here: waypoint_extraction.traj_reconstruction
imports `robosuite.utils.transform_utils` purely for quaternion math, but
importing `robosuite` at all eagerly runs its full mujoco/EGL rendering
stack (robosuite/__init__.py -> environments -> binding_utils ->
egl_context), which crashes on machines without a working EGL/OSMesa OpenGL
backend -- even though dp/greedy waypoint selection never render anything
(env=None). _import_waypoint_extraction() pre-registers a stub top-level
`robosuite` module (pointing __path__ at the real installed package, so
submodule imports still resolve to the real files) so Python skips
robosuite's top-level __init__.py; only the leaf modules actually needed
(transform_utils, numba, macros, log_utils -- pure math/logging, no
mujoco/EGL) get imported for real.

Known upstream bug: waypoint_extraction's pos_only reconstruction path
(pos_only_geometric_waypoint_trajectory) expects `gt_states` as raw position
arrays, but the non-pos_only path (used to build gt_states here) needs it as
a list of {"robot0_eef_pos", "robot0_eef_quat"} dicts. Passing pos_only=True
therefore raises a TypeError inside the library -- not something fixable
from this side without forking the package.

Second packaging quirk: waypoint_extraction.traj_reconstruction does
`from utils import put_text, remove_object` -- an ABSOLUTE import of a
top-level `utils` module that the pip package does not itself ship (it
assumes it's run from inside the original AWE repo checkout, sitting next
to a sibling utils.py with cv2/mujoco debug-rendering helpers). Whatever
`utils` package happens to be first on sys.path silently satisfies that
import instead -- in this repo that's mimicgen/mimicgen/utils/, which has no
put_text/remove_object and errors. Neither symbol is ever called by
dp_waypoint_selection/greedy_waypoint_selection (they're only used by
traj_reconstruction's video-annotation helpers, unreachable from the
geometric-only, env=None path this module uses), so
_import_waypoint_extraction() pre-registers a stub `utils` module in
sys.modules with no-op put_text/remove_object before importing
waypoint_extraction, exactly like the robosuite stub above.
"""

import importlib.util
import sys
import types

import numpy as np


def _stub_utils_module():
    """Satisfy waypoint_extraction.traj_reconstruction's `from utils import
    put_text, remove_object` with no-op stand-ins, so a same-named local
    package elsewhere on sys.path (e.g. mimicgen/utils/) can't shadow it and
    break an import path that's unreachable from this module's actual usage
    anyway. See module docstring for the full explanation."""
    existing = sys.modules.get("utils")
    if existing is not None and hasattr(existing, "put_text") and hasattr(existing, "remove_object"):
        return
    stub = types.ModuleType("utils")
    stub.put_text = lambda *a, **k: None
    stub.remove_object = lambda *a, **k: None
    sys.modules["utils"] = stub


def _import_waypoint_extraction():
    if "robosuite" not in sys.modules:
        try:
            import robosuite  # noqa: F401
        except AttributeError:
            for name in list(sys.modules):
                if name == "robosuite" or name.startswith("robosuite."):
                    del sys.modules[name]
            spec = importlib.util.find_spec("robosuite")
            stub = types.ModuleType("robosuite")
            stub.__path__ = spec.submodule_search_locations
            sys.modules["robosuite"] = stub

    _stub_utils_module()

    from waypoint_extraction import dp_waypoint_selection, greedy_waypoint_selection

    return dp_waypoint_selection, greedy_waypoint_selection


dp_waypoint_selection, greedy_waypoint_selection = _import_waypoint_extraction()


VALID_METHODS = ("greedy", "dp")


def _select_waypoints(  # noqa: PLR0913, PLR0917
    eef_pos, eef_quat, gripper_cmd, err_threshold, method, pos_only
):
    """Run AWE over one demo's trajectory and return a sorted list of
    frame indices (0-indexed, always including the last frame)."""
    num_frames = eef_pos.shape[0]

    # `actions[:, :3]` supplies the waypoint positions for geometric interpolation;
    # `actions[:, -1]` supplies the gripper open/close toggle signal. Passing the
    # eef position itself (rather than a delta action) is correct here since AWE's
    # geometric error is computed against absolute end-effector positions.
    actions = np.concatenate([eef_pos, gripper_cmd[:, None]], axis=1)  # (T, 4)
    gt_states = [
        {"robot0_eef_pos": eef_pos[t], "robot0_eef_quat": eef_quat[t]}
        for t in range(num_frames)
    ]

    if method == "dp":
        waypoints = dp_waypoint_selection(
            env=None,
            actions=actions,
            gt_states=gt_states,
            err_threshold=err_threshold,
            pos_only=pos_only,
        )
    elif method == "greedy":
        waypoints = greedy_waypoint_selection(
            env=None,
            actions=actions,
            gt_states=gt_states,
            err_threshold=err_threshold,
            geometry=True,  # stay on the geometric (no-simulator) path
            pos_only=pos_only,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Valid: {VALID_METHODS}")

    # Drop a frame-0 waypoint (possible if the gripper command flips between
    # frames 0 and 1) so the switch-index convention matches
    # rdp_subgoal_decomp.py/bspline_subgoal_decomp.py's _finalize_indices,
    # which always excludes index 0 -- a boundary there is degenerate (goal
    # == the gripper's own frame-0 pose) and would otherwise make AWE the
    # only method capable of producing it, skewing method comparisons.
    return sorted({int(w) for w in waypoints if w > 0})


def _assign_subgoals(waypoints, gripper_pcd):
    """For every frame, assign the gripper keypoints of the next upcoming
    waypoint (inclusive) as its subgoal -- goals repeat until the demo passes
    each waypoint, matching subgoal_decomp.py/rdp_subgoal_decomp.py's
    goal_gripper_pcd convention."""
    num_frames = gripper_pcd.shape[0]
    goals = np.zeros((num_frames, 4, 3), dtype=np.float32)
    wp_idx = 0
    for t in range(num_frames):
        while waypoints[wp_idx] < t:
            wp_idx += 1
        goals[t] = gripper_pcd[waypoints[wp_idx]]
    return goals


def compute_awe_subgoal_gripper_pcd(  # noqa: PLR0913, PLR0917
    gripper_pcd: np.ndarray,  # (T, 4, 3)
    eef_pos: np.ndarray,  # (T, 3)
    eef_quat: np.ndarray,  # (T, 4)  (x, y, z, w)
    actions: np.ndarray,  # (T, D)  last dim = gripper open/close command
    err_threshold: float = 0.01,
    method: str = "greedy",  # "greedy" | "dp"
    pos_only: bool = False,
    return_switch_idxs: bool = False,
):
    """
    Compute a goal_gripper_pcd via AWE (Automatic Waypoint Extraction).

    Returns:
        expanded_goal_gripper_pcd: (T, 4, 3) float32
        switch_indices (optional): sorted list[int] of waypoints (ends at T-1)
    """
    gripper_cmd = np.asarray(actions, dtype=np.float64)[:, -1]
    waypoints = _select_waypoints(
        np.asarray(eef_pos, dtype=np.float64),
        np.asarray(eef_quat, dtype=np.float64),
        gripper_cmd,
        err_threshold,
        method,
        pos_only,
    )
    expanded = _assign_subgoals(waypoints, gripper_pcd)

    if return_switch_idxs:
        return expanded, waypoints
    return expanded
