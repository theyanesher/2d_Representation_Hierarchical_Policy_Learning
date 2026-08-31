import contextlib
import io
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOGEN = ROOT / "third_party" / "robogen"
sys.path.insert(0, str(ROBOGEN))

import awe_subgoal_decomp as awe  # noqa: E402
from waypoint_extraction import traj_reconstruction  # noqa: E402


class SafePointLineDistanceTests(unittest.TestCase):
    def test_installed_helper_is_patched(self):
        self.assertIs(
            traj_reconstruction.point_line_distance,
            awe._safe_point_line_distance,
        )

    def test_zero_length_segment_uses_distance_to_endpoint(self):
        point = np.array([1.0, 2.0, 2.0])
        endpoint = np.zeros(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            distance = traj_reconstruction.point_line_distance(
                point, endpoint, endpoint
            )
        self.assertEqual(distance, 3.0)

    def test_greedy_handles_stationary_span_at_gripper_transition(self):
        eef_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )
        eef_quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(eef_pos), 1))
        gripper_cmd = np.array([-1.0, -1.0, 1.0, 1.0])

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("error", RuntimeWarning)
            waypoints = awe._select_waypoints(
                eef_pos,
                eef_quat,
                gripper_cmd,
                err_threshold=0.1,
                method="greedy",
                pos_only=False,
                # this test is specifically about the gripper-transition
                # seeding path, which is opt-in since it defaults to off
                use_gripper_seeding=True,
            )

        self.assertEqual(waypoints, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
