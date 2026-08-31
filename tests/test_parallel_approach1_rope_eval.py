import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = (
    ROOT
    / "external/mimicgen/mimicgen/scripts/eval_gmm_high_level_rope_2d_dit_low_level.py"
)
SPEC = importlib.util.spec_from_file_location("parallel_approach1_rope_eval", EVAL_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def _fake_observations(n_envs, height=4, width=5):
    shape_meta = EVAL.get_shape_meta(height, width)
    obs = {
        key: np.zeros((n_envs,) + tuple(meta["shape"]), dtype=np.float32)
        for key, meta in shape_meta["obs"].items()
    }
    obs["robot0_eef_quat"][..., 3] = 1.0
    return obs


class BatchedInputTests(unittest.TestCase):
    def test_hl_and_ll_inputs_select_active_slots_and_repeat_observation(self):
        obs = _fake_observations(3)
        obs["point_cloud"][:, :, 0] = np.arange(3)[:, None]
        obs["agentview_image"][:, 0, 0, 0] = [10, 20, 30]

        scene, gripper = EVAL.build_batched_hl_inputs_lfd3d(
            obs, active_indices=[2, 0], device="cpu"
        )
        ll_obs = EVAL.build_batched_ll_obs_dict(
            obs, active_indices=[2, 0], n_obs_steps=2, device="cpu"
        )

        self.assertEqual(tuple(scene.shape), (2, 4500, 3))
        self.assertEqual(tuple(gripper.shape), (2, 4, 3))
        torch.testing.assert_close(scene[:, 0, 0], torch.tensor([2.0, 0.0]))
        self.assertEqual(tuple(ll_obs["cam0_image"].shape), (2, 2, 3, 4, 5))
        self.assertEqual(tuple(ll_obs["cam0_intrinsic"].shape), (2, 2, 3, 3))
        torch.testing.assert_close(
            ll_obs["cam0_image"][:, 0], ll_obs["cam0_image"][:, 1]
        )


class _FakeLegacyPolicy:
    action_horizon = 3
    action_dim = 10
    dtype = torch.float32

    def __init__(self):
        self.seen_noise = None

    def predict_action(self, obs):
        batch = obs["state"].shape[0]
        self.seen_noise = torch.randn(
            batch, self.action_horizon, self.action_dim,
            dtype=self.dtype, device=obs["state"].device,
        )
        return {"action_pred": self.seen_noise}


class NoiseCompatibilityTests(unittest.TestCase):
    def test_legacy_policy_receives_exact_per_episode_noise(self):
        policy = _FakeLegacyPolicy()
        generators = EVAL._make_episode_generators([101, 202], device="cpu")
        expected = EVAL._sample_episode_initial_noise(
            policy, generators, active_indices=[0, 1], device="cpu"
        )
        obs = {"state": torch.zeros(2, 2, 10)}

        EVAL._predict_action_with_initial_noise(policy, obs, expected)

        torch.testing.assert_close(policy.seen_noise, expected, rtol=0, atol=0)

    def test_noise_stream_is_independent_of_sibling_completion(self):
        policy = _FakeLegacyPolicy()
        generators = EVAL._make_episode_generators([11, 22, 33], device="cpu")
        first = EVAL._sample_episode_initial_noise(
            policy, generators, active_indices=[0, 1, 2], device="cpu"
        )
        second = EVAL._sample_episode_initial_noise(
            policy, generators, active_indices=[0, 2], device="cpu"
        )

        serial_zero = EVAL._make_episode_generators([11], device="cpu")
        serial_two = EVAL._make_episode_generators([33], device="cpu")
        expected_zero_1 = EVAL._sample_episode_initial_noise(
            policy, serial_zero, active_indices=[0], device="cpu"
        )[0]
        expected_zero_2 = EVAL._sample_episode_initial_noise(
            policy, serial_zero, active_indices=[0], device="cpu"
        )[0]
        expected_two_1 = EVAL._sample_episode_initial_noise(
            policy, serial_two, active_indices=[0], device="cpu"
        )[0]
        expected_two_2 = EVAL._sample_episode_initial_noise(
            policy, serial_two, active_indices=[0], device="cpu"
        )[0]

        torch.testing.assert_close(first[0], expected_zero_1, rtol=0, atol=0)
        torch.testing.assert_close(second[0], expected_zero_2, rtol=0, atol=0)
        torch.testing.assert_close(first[2], expected_two_1, rtol=0, atol=0)
        torch.testing.assert_close(second[1], expected_two_2, rtol=0, atol=0)


class _FakeHighLevel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.data_sources = None

    def forward(self, net_in, text_embedding, data_source):
        self.data_sources = data_source
        batch, _, anchors = net_in.shape
        return torch.zeros(batch, anchors, 13, dtype=net_in.dtype)


class _FakeVectorEnv:
    def __init__(self, n_envs):
        self.num_envs = n_envs
        self.obs = _fake_observations(n_envs)
        self.config = None
        self.progress = np.zeros(n_envs, dtype=int)
        self.done_after = np.ones(n_envs, dtype=int)

    def call(self, name):
        if name == "get_controller_limits":
            return [(0.05, 0.5)] * self.num_envs
        if name == "get_episode_result":
            return [
                {
                    "reward": float(seed),
                    "success": bool(seed % 2),
                    "steps": int(self.progress[i]),
                    "video": None,
                    "video_seconds": 0.0,
                }
                for i, (seed, _video, _active) in enumerate(self.config)
            ]
        raise AssertionError(name)

    def call_each(self, name, args_list):
        self.assert_name = name
        self.config = args_list
        return [None] * self.num_envs

    def reset(self):
        self.progress[:] = 0
        for slot, (seed, _video, active) in enumerate(self.config):
            self.done_after[slot] = (seed % 3) + 1 if active else 0
            self.obs["state"][slot, 0] = seed
        return self.obs

    def step(self, actions):
        done = np.zeros(self.num_envs, dtype=bool)
        for slot, (_seed, _video, active) in enumerate(self.config):
            if not active:
                done[slot] = True
                continue
            self.progress[slot] += 1
            done[slot] = self.progress[slot] >= self.done_after[slot]
        return self.obs, np.zeros(self.num_envs), done, ({},) * self.num_envs


class _FakeRolloutPolicy(_FakeLegacyPolicy):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def predict_action(self, obs):
        batch = obs["state"].shape[0]
        self.batch_sizes.append(batch)
        self.seen_noise = torch.randn(
            batch, self.action_horizon, self.action_dim,
            dtype=self.dtype, device=obs["state"].device,
        )
        action = torch.zeros_like(self.seen_noise)
        action[..., 3] = 1.0
        action[..., 7] = 1.0
        return {"action_pred": action}


class _LazyPointCloudEnv:
    def __init__(self, succeed_after=None):
        self.enabled = True
        self.step_flags = []
        self.step_count = 0
        self.succeed_after = succeed_after

    def set_point_cloud_enabled(self, enabled=True):
        self.enabled = bool(enabled)

    def _obs(self):
        obs = {"agentview_image": np.zeros((3, 4, 5), dtype=np.float32)}
        if self.enabled:
            obs["point_cloud"] = np.zeros((4500, 3), dtype=np.float32)
        return obs

    def step(self, action):
        self.step_flags.append(self.enabled)
        self.step_count += 1
        return self._obs(), 0.0, False, {}

    def is_success(self):
        success = self.succeed_after is not None and self.step_count >= self.succeed_after
        return {"task": success}

    def get_observation(self):
        return self._obs()


def _make_lazy_worker(fake_env):
    worker = EVAL._ParallelMimicGenEpisodeEnv.__new__(
        EVAL._ParallelMimicGenEpisodeEnv
    )
    worker.env = fake_env
    worker.observation_space = {"agentview_image": None, "point_cloud": None}
    worker.last_obs = fake_env._obs()
    worker.done = False
    worker.success = False
    worker.steps = 0
    worker.max_steps = 800
    worker.total_reward = 0.0
    worker.video_recorder = None
    worker.video_seconds = 0.0
    worker.point_cloud_builds = 0
    worker.point_cloud_skips = 0
    return worker


class HighLevelBatchTests(unittest.TestCase):
    def test_gmm_inference_supports_batch_and_normalizes_each_sample(self):
        model = _FakeHighLevel()
        scene = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        gripper = torch.zeros(2, 4, 3)
        text = torch.zeros(2, 1152)

        means, probabilities = EVAL.infer_articubot_gmm(
            model, scene, gripper, text
        )

        self.assertEqual(tuple(means.shape), (2, 5, 4, 3))
        self.assertEqual(tuple(probabilities.shape), (2, 5))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
        self.assertEqual(model.data_sources, ["libero_franka", "libero_franka"])


class ParallelSchedulerTests(unittest.TestCase):
    def test_partial_chunks_batch_both_policies_and_preserve_episode_order(self):
        env = _FakeVectorEnv(n_envs=3)
        hl_model = _FakeHighLevel()
        ll_model = _FakeRolloutPolicy()
        output = io.StringIO()

        rewards, successes, timing = EVAL.run_parallel_episodes(
            env,
            hl_model,
            ll_model,
            torch.zeros(1, 1152),
            n_episodes=5,
            seed=10,
            n_obs_steps=2,
            n_action_steps=2,
            device="cpu",
            inference_dtype="fp32",
            save_videos=False,
            num_video_episodes=0,
            videos_dir=Path("unused"),
            results_f=output,
        )

        self.assertEqual(rewards, [10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual(successes, [False, True, False, True, False])
        self.assertIn(3, ll_model.batch_sizes)
        self.assertIn(1, ll_model.batch_sizes)
        self.assertGreater(timing["mean_policy_batch_size"], 1)
        self.assertGreater(timing["episodes_per_hour"], 0)
        self.assertEqual(len(output.getvalue().splitlines()), 5)


class LazyPointCloudTests(unittest.TestCase):
    def test_point_cloud_is_built_only_on_action_chunk_boundary(self):
        fake_env = _LazyPointCloudEnv()
        worker = _make_lazy_worker(fake_env)

        worker.step(np.zeros((8, 7), dtype=np.float32))

        self.assertEqual(fake_env.step_flags, [False] * 7 + [True])
        self.assertEqual(worker.point_cloud_skips, 7)
        self.assertEqual(worker.point_cloud_builds, 1)
        self.assertIn("point_cloud", worker.last_obs)

    def test_early_success_rebuilds_fixed_schema_observation(self):
        fake_env = _LazyPointCloudEnv(succeed_after=2)
        worker = _make_lazy_worker(fake_env)

        obs, _reward, done, _info = worker.step(
            np.zeros((8, 7), dtype=np.float32)
        )

        self.assertTrue(done)
        self.assertEqual(fake_env.step_flags, [False, False])
        self.assertEqual(worker.point_cloud_skips, 2)
        self.assertEqual(worker.point_cloud_builds, 1)
        self.assertIn("point_cloud", obs)


if __name__ == "__main__":
    unittest.main()
