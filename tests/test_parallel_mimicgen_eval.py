import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = (
    ROOT
    / "external/mimicgen/mimicgen/scripts/eval_approach2_from_gmm_2d_dit_low_level.py"
)
SPEC = importlib.util.spec_from_file_location("parallel_mimicgen_eval", EVAL_PATH)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class BatchedInputTests(unittest.TestCase):
    def test_repeats_current_observation_without_changing_values(self):
        shape_meta = EVAL.get_shape_meta(4, 5)
        obs = {
            key: np.arange(3 * np.prod(meta["shape"]), dtype=np.float32).reshape(
                (3,) + tuple(meta["shape"])
            )
            for key, meta in shape_meta["obs"].items()
        }
        result = EVAL.build_batched_ll_obs_dict(
            obs, active_indices=[2, 0], n_obs_steps=2, device="cpu"
        )

        self.assertEqual(result["cam0_image"].shape, (2, 2, 3, 4, 5))
        self.assertEqual(result["present_gripper_pts"].shape, (2, 2, 4, 3))
        self.assertNotIn("point_cloud", shape_meta["obs"])
        torch.testing.assert_close(result["cam0_image"][:, 0], result["cam0_image"][:, 1])
        torch.testing.assert_close(
            result["state"][:, 0], torch.from_numpy(obs["state"][[2, 0]])
        )


class ActionConversionTests(unittest.TestCase):
    def test_vectorized_conversion_matches_scalar_rotation_math(self):
        rng = np.random.default_rng(7)
        batch, horizon = 4, 6
        actions = rng.normal(scale=0.02, size=(batch, horizon, 10))
        actions[..., 3:9] += np.array([1, 0, 0, 0, 1, 0])
        quats = Rotation.random(batch, random_state=rng).as_quat()

        actual = EVAL.policy_action_batch_to_env_action_vectorized(
            actions, quats, max_dpos=0.05, max_drot=0.5
        )
        expected = np.empty((batch, horizon, 7), dtype=np.float32)
        for b in range(batch):
            cur = Rotation.from_quat(quats[b]).as_matrix()
            for t in range(horizon):
                a1, a2 = actions[b, t, 3:9].reshape(2, 3)
                b1 = a1 / np.linalg.norm(a1)
                b2 = a2 - np.dot(a2, b1) * b1
                b2 = b2 / np.linalg.norm(b2)
                delta = np.stack((b1, b2, np.cross(b1, b2)), axis=1)
                rotvec = Rotation.from_matrix(cur @ delta @ cur.T).as_rotvec()
                expected[b, t, :3] = np.clip(actions[b, t, :3] / 0.05, -1, 1)
                expected[b, t, 3:6] = np.clip(rotvec / 0.5, -1, 1)
                expected[b, t, 6] = np.clip(actions[b, t, 9] / -0.01, -1, 1)

        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def _fake_observations(n_envs):
    shape_meta = EVAL.get_shape_meta(4, 5)
    result = {}
    for key, meta in shape_meta["obs"].items():
        value = np.zeros((n_envs,) + tuple(meta["shape"]), dtype=np.float32)
        result[key] = value
    result["robot0_eef_quat"][..., 3] = 1.0
    return result


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
        for i, (seed, _video, active) in enumerate(self.config):
            self.done_after[i] = (seed % 3) + 1 if active else 0
            self.obs["state"][i, 0] = seed
        return self.obs

    def step(self, actions):
        done = np.zeros(self.num_envs, dtype=bool)
        for i, (_seed, _video, active) in enumerate(self.config):
            if not active:
                done[i] = True
                continue
            self.progress[i] += 1
            done[i] = self.progress[i] >= self.done_after[i]
        return self.obs, np.zeros(self.num_envs), done, ({},) * self.num_envs


class _FakePolicy:
    def __init__(self, horizon):
        self.horizon = horizon
        self.action_horizon = horizon
        self.action_dim = 10
        self.dtype = torch.float32
        self.batch_sizes = []
        self.noise_by_seed = {}

    def predict_action(self, obs, initial_noise=None):
        batch = obs["state"].shape[0]
        self.batch_sizes.append(batch)
        self.assert_initial_noise_shape(initial_noise, batch)
        for seed, noise in zip(obs["state"][:, 0, 0].tolist(), initial_noise):
            self.noise_by_seed.setdefault(int(seed), []).append(noise.clone())
        action = torch.zeros((batch, self.horizon, 10), dtype=torch.float32)
        action[..., 3] = 1.0
        action[..., 7] = 1.0
        return {"action_pred": action}

    def assert_initial_noise_shape(self, initial_noise, batch):
        if initial_noise is None:
            raise AssertionError("parallel evaluator did not supply initial noise")
        expected = (batch, self.action_horizon, self.action_dim)
        if tuple(initial_noise.shape) != expected:
            raise AssertionError(
                f"initial noise shape {tuple(initial_noise.shape)} != {expected}"
            )


def _make_spawn_smoke_env():
    from gym import spaces

    class SpawnSmokeEnv:
        metadata = {}

        def __init__(self):
            self.observation_space = spaces.Dict(
                {"value": spaces.Box(-10, 10, shape=(2,), dtype=np.float32)}
            )
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
            self.value = np.zeros(2, dtype=np.float32)

        def reset(self):
            self.value[:] = 0
            return {"value": self.value.copy()}

        def step(self, action):
            self.value += float(action[0])
            return {"value": self.value.copy()}, 0.0, False, {}

        def seed(self, seed=None):
            return None

        def close(self):
            return None

    return SpawnSmokeEnv()


class ParallelSchedulerTests(unittest.TestCase):
    def _run_fake_eval(self, n_envs):
        env = _FakeVectorEnv(n_envs=n_envs)
        policy = _FakePolicy(horizon=2)
        EVAL.run_parallel_episodes(
            env,
            policy,
            n_episodes=5,
            seed=10,
            n_obs_steps=2,
            n_action_steps=2,
            device="cpu",
            inference_dtype="fp32",
            save_videos=False,
            num_video_episodes=0,
            videos_dir=Path("unused"),
            video_fps=10,
            results_f=io.StringIO(),
        )
        return policy

    def test_repository_async_vector_env_uses_spawn_and_shared_memory(self):
        from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv

        env = AsyncVectorEnv(
            [_make_spawn_smoke_env, _make_spawn_smoke_env],
            shared_memory=True,
            copy=False,
            context="spawn",
        )
        try:
            obs = env.reset()
            np.testing.assert_array_equal(obs["value"], np.zeros((2, 2)))
            obs, _reward, _done, _info = env.step(
                np.array([[0.5], [-0.25]], dtype=np.float32)
            )
            np.testing.assert_allclose(obs["value"][:, 0], [0.5, -0.25])
        finally:
            env.close()

    def test_partial_chunks_preserve_episode_order_and_active_batching(self):
        env = _FakeVectorEnv(n_envs=3)
        policy = _FakePolicy(horizon=2)
        output = io.StringIO()
        rewards, successes, timing = EVAL.run_parallel_episodes(
            env,
            policy,
            n_episodes=5,
            seed=10,
            n_obs_steps=2,
            n_action_steps=2,
            device="cpu",
            inference_dtype="fp32",
            save_videos=False,
            num_video_episodes=0,
            videos_dir=Path("unused"),
            video_fps=10,
            results_f=output,
        )

        self.assertEqual(rewards, [10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual(successes, [False, True, False, True, False])
        self.assertEqual([line.count('"episode"') for line in output.getvalue().splitlines()], [1] * 5)
        self.assertIn(3, policy.batch_sizes)
        self.assertIn(1, policy.batch_sizes)
        self.assertGreater(timing["mean_policy_batch_size"], 1)
        self.assertGreater(timing["episodes_per_hour"], 0)

    def test_episode_noise_is_invariant_to_batch_size_and_completion_order(self):
        serial = self._run_fake_eval(n_envs=1)
        parallel = self._run_fake_eval(n_envs=3)

        self.assertEqual(serial.noise_by_seed.keys(), parallel.noise_by_seed.keys())
        for seed in serial.noise_by_seed:
            serial_draws = serial.noise_by_seed[seed]
            parallel_draws = parallel.noise_by_seed[seed]
            self.assertEqual(len(serial_draws), len(parallel_draws))
            for serial_draw, parallel_draw in zip(serial_draws, parallel_draws):
                torch.testing.assert_close(serial_draw, parallel_draw, rtol=0, atol=0)

    def test_episode_generator_matches_original_serial_rng_stream(self):
        policy = _FakePolicy(horizon=2)
        episode_seed = 12345
        generators = EVAL._make_episode_generators([episode_seed], device="cpu")

        for _ in range(3):
            actual = EVAL._sample_episode_initial_noise(
                policy, generators, active_indices=[0], device="cpu"
            )

            if _ == 0:
                torch.manual_seed(episode_seed)
            expected = torch.randn(
                (1, policy.action_horizon, policy.action_dim),
                dtype=policy.dtype,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
