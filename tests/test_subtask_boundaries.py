import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from subtask_boundaries import (
    BoundaryPrediction,
    GeminiBoundaryVLM,
    OpenAIBoundaryVLM,
    QwenBoundaryVLM,
    RefinedBoundary,
    detect_subtask_boundaries,
    load_rgb_npz,
    make_contact_sheets,
    normalize_rgb_frames,
    refinement_frame_indices,
    sample_frame_indices,
    sanitize_boundary_indices,
)
from subtask_boundaries.contact_sheet import _frame_label


class FakeVLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def predict(self, *, prompt, contact_sheets, schema):
        self.calls.append((prompt, contact_sheets, schema))
        response = self.responses.pop(0)
        return schema.model_validate(response)


class InputHandlingTests(unittest.TestCase):
    def test_channel_last_uint8(self):
        frames = np.zeros((4, 12, 16, 3), dtype=np.uint8)
        converted = normalize_rgb_frames(frames)
        self.assertEqual(converted.shape, (4, 12, 16, 3))
        self.assertEqual(converted.dtype, np.uint8)

    def test_channel_first_is_converted(self):
        frames = np.zeros((4, 3, 12, 16), dtype=np.uint8)
        converted = normalize_rgb_frames(frames)
        self.assertEqual(converted.shape, (4, 12, 16, 3))

    def test_float_unit_range_is_scaled_safely(self):
        frames = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)
        converted = normalize_rgb_frames(frames)
        np.testing.assert_array_equal(converted[0, 0, 0], [0, 128, 255])

    def test_invalid_shape_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, "4D array"):
            normalize_rgb_frames(np.zeros((10, 20, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "exactly 3 channels"):
            normalize_rgb_frames(np.zeros((2, 10, 20, 4), dtype=np.uint8))

    def test_episode_npz_loader(self):
        frames = np.zeros((3, 8, 9, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            np.savez(path, rgb=frames)
            loaded = load_rgb_npz(path)
        np.testing.assert_array_equal(loaded, frames)


class IndexAndSheetTests(unittest.TestCase):
    def test_sampling_keeps_original_indices(self):
        self.assertEqual(
            sample_frame_indices(100, 15), [0, 15, 30, 45, 60, 75, 90, 99]
        )

    def test_sampling_does_not_duplicate_aligned_terminal_index(self):
        self.assertEqual(sample_frame_indices(91, 15), [0, 15, 30, 45, 60, 75, 90])

    def test_contact_sheet_records_and_renders_original_index(self):
        frames = np.full((40, 10, 20, 3), 255, dtype=np.uint8)
        sheets = make_contact_sheets(
            frames, [0, 15, 30], frame_width=40, frames_per_sheet=4, columns=2
        )
        self.assertEqual(sheets[0].frame_indices, (0, 15, 30))
        self.assertEqual(_frame_label(30), "FRAME 30")
        # The original frame is white, so this black pixel proves the label
        # badge was painted into the image tile rather than supplied as text.
        self.assertEqual(sheets[0].image.getpixel((0, 0)), (0, 0, 0))

    def test_consecutive_sheets_overlap_by_original_indices(self):
        frames = np.zeros((6, 10, 20, 3), dtype=np.uint8)
        sheets = make_contact_sheets(
            frames,
            [0, 1, 2, 3, 4, 5],
            frame_width=40,
            frames_per_sheet=4,
            columns=2,
            sheet_overlap_frames=2,
        )
        self.assertEqual(
            [sheet.frame_indices for sheet in sheets],
            [(0, 1, 2, 3), (2, 3, 4, 5)],
        )

    def test_partial_final_sheet_uses_only_populated_rows(self):
        frames = np.zeros((5, 10, 20, 3), dtype=np.uint8)
        sheets = make_contact_sheets(
            frames,
            [0, 1, 2, 3, 4],
            frame_width=40,
            frames_per_sheet=4,
            columns=2,
            sheet_overlap_frames=1,
        )
        self.assertEqual(sheets[0].image.size, (80, 40))
        self.assertEqual(sheets[1].frame_indices, (3, 4))
        self.assertEqual(sheets[1].image.size, (80, 20))

    def test_boundary_sanitation_clamps_sorts_and_deduplicates(self):
        result = sanitize_boundary_indices([120, 45, 120, 999, -10], 300)
        self.assertEqual(result, [0, 45, 120, 299])

    def test_conservative_distance_merge_keeps_earlier_boundary(self):
        result = sanitize_boundary_indices(
            [117, 119, 205], 300, min_boundary_distance_frames=5
        )
        self.assertEqual(result, [117, 205])


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.zeros((40, 12, 16, 3), dtype=np.uint8)

    def test_empty_coarse_output_is_valid(self):
        vlm = FakeVLM({"boundary_indices": []})
        result = detect_subtask_boundaries(
            self.frames, vlm=vlm, refine=True, logs_dir=None
        )
        self.assertEqual(result, [])
        self.assertEqual(len(vlm.calls), 1)
        self.assertIs(vlm.calls[0][2], BoundaryPrediction)

    def test_stop_after_sparse_annotation_skips_dense_calls(self):
        vlm = FakeVLM({"boundary_indices": [15]})
        result = detect_subtask_boundaries(
            self.frames,
            vlm=vlm,
            sample_every_n_frames=15,
            refine=True,
            stop_after_sparse_annotation=True,
            logs_dir=None,
        )
        self.assertEqual(result, [15])
        self.assertEqual(len(vlm.calls), 1)
        self.assertIs(vlm.calls[0][2], BoundaryPrediction)

    def test_dense_refinement_returns_original_index(self):
        vlm = FakeVLM(
            {"boundary_indices": [15]},
            {"boundary_index": 17},
        )
        result = detect_subtask_boundaries(
            self.frames,
            vlm=vlm,
            sample_every_n_frames=15,
            refinement_radius=3,
            refinement_stride=1,
            logs_dir=None,
        )
        self.assertEqual(result, [17])
        self.assertIs(vlm.calls[1][2], RefinedBoundary)
        visible = tuple(
            index
            for sheet in vlm.calls[1][1]
            for index in sheet.frame_indices
        )
        self.assertEqual(visible, tuple(range(12, 19)))

    def test_coarse_output_must_be_a_visibly_sampled_original_index(self):
        vlm = FakeVLM({"boundary_indices": [14, 15, 999]})
        result = detect_subtask_boundaries(
            self.frames,
            vlm=vlm,
            sample_every_n_frames=15,
            refine=False,
            logs_dir=None,
        )
        self.assertEqual(result, [15])

    def test_refinement_windows_clamp_at_episode_edges(self):
        self.assertEqual(refinement_frame_indices(10, 0, 3, 1), [0, 1, 2, 3])
        self.assertEqual(refinement_frame_indices(10, 9, 3, 1), [6, 7, 8, 9])

    def test_openai_adapter_sends_images_with_structured_schema(self):
        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def parse(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    output_parsed=BoundaryPrediction(boundary_indices=[15])
                )

        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        backend = OpenAIBoundaryVLM(client=client)
        sheets = make_contact_sheets(
            self.frames,
            [0, 15],
            frame_width=32,
            frames_per_sheet=2,
            columns=2,
            sheet_overlap_frames=0,
        )
        result = backend.predict(
            prompt="find boundaries",
            contact_sheets=sheets,
            schema=BoundaryPrediction,
        )
        self.assertEqual(result.boundary_indices, [15])
        self.assertIs(responses.kwargs["text_format"], BoundaryPrediction)
        image_part = responses.kwargs["input"][0]["content"][1]
        self.assertEqual(image_part["type"], "input_image")
        self.assertTrue(image_part["image_url"].startswith("data:image/jpeg;base64,"))

    def test_gemini_adapter_sends_images_with_structured_schema(self):
        class FakeInteractions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(output_text='{"boundary_indices":[15]}')

        interactions = FakeInteractions()
        client = SimpleNamespace(interactions=interactions)
        backend = GeminiBoundaryVLM(client=client)
        sheets = make_contact_sheets(
            self.frames,
            [0, 15],
            frame_width=32,
            frames_per_sheet=2,
            columns=2,
            sheet_overlap_frames=0,
        )
        result = backend.predict(
            prompt="find boundaries",
            contact_sheets=sheets,
            schema=BoundaryPrediction,
        )
        self.assertEqual(result.boundary_indices, [15])
        self.assertEqual(interactions.kwargs["model"], "gemini-3.5-flash")
        response_format = interactions.kwargs["response_format"]
        self.assertEqual(response_format["mime_type"], "application/json")
        self.assertEqual(response_format["schema"]["type"], "object")
        image_part = interactions.kwargs["input"][1]
        self.assertEqual(image_part["type"], "image")
        self.assertEqual(image_part["mime_type"], "image/jpeg")
        self.assertNotIn("data:image", image_part["data"])

    def test_gemini_provider_selects_3_5_flash_by_default(self):
        fake_backend = FakeVLM({"boundary_indices": []})
        with patch(
            "subtask_boundaries.detector.GeminiBoundaryVLM",
            return_value=fake_backend,
        ) as constructor:
            result = detect_subtask_boundaries(
                self.frames, provider="gemini", refine=False, logs_dir=None
            )
        self.assertEqual(result, [])
        constructor.assert_called_once_with(model="gemini-3.5-flash")

    def test_local_qwen_adapter_sends_images_with_structured_schema(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def parse(self, **kwargs):
                self.kwargs = kwargs
                message = SimpleNamespace(
                    parsed=BoundaryPrediction(boundary_indices=[15])
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        backend = QwenBoundaryVLM(client=client)
        sheets = make_contact_sheets(
            self.frames,
            [0, 15],
            frame_width=32,
            frames_per_sheet=2,
            columns=2,
            sheet_overlap_frames=0,
        )
        result = backend.predict(
            prompt="find boundaries",
            contact_sheets=sheets,
            schema=BoundaryPrediction,
        )
        self.assertEqual(result.boundary_indices, [15])
        self.assertEqual(completions.kwargs["model"], "qwen3.6-local")
        self.assertIs(completions.kwargs["response_format"], BoundaryPrediction)
        self.assertEqual(
            completions.kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(completions.kwargs["max_completion_tokens"], 256)
        self.assertEqual(completions.kwargs["temperature"], 0)
        image_part = completions.kwargs["messages"][0]["content"][0]
        self.assertEqual(image_part["type"], "image_url")
        self.assertTrue(
            image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_local_qwen_uses_loopback_without_an_api_key(self):
        with patch("openai.OpenAI") as constructor:
            QwenBoundaryVLM()
        constructor.assert_called_once_with(
            api_key="local", base_url="http://127.0.0.1:8000/v1"
        )

    def test_qwen_is_default_and_selects_local_served_model(self):
        fake_backend = FakeVLM({"boundary_indices": []})
        with patch(
            "subtask_boundaries.detector.QwenBoundaryVLM",
            return_value=fake_backend,
        ) as constructor:
            result = detect_subtask_boundaries(
                self.frames, refine=False, logs_dir=None
            )
        self.assertEqual(result, [])
        constructor.assert_called_once_with(model="qwen3.6-local", base_url=None)

    def test_sparse_input_and_output_are_written_to_unique_log_dir(self):
        vlm = FakeVLM({"boundary_indices": [15]})
        with tempfile.TemporaryDirectory() as directory:
            result = detect_subtask_boundaries(
                self.frames,
                vlm=vlm,
                sample_every_n_frames=15,
                stop_after_sparse_annotation=True,
                logs_dir=directory,
            )
            runs = list(Path(directory).iterdir())
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            self.assertTrue((run_dir / "prompt.txt").is_file())
            self.assertTrue((run_dir / "contact_sheet_001.jpg").is_file())
            logged_input = json.loads((run_dir / "input.json").read_text())
            logged_output = json.loads((run_dir / "output.json").read_text())

        self.assertEqual(result, [15])
        self.assertEqual(
            logged_input["sampled_original_frame_indices"], [0, 15, 30, 39]
        )
        self.assertEqual(logged_input["sheet_overlap_frames"], 2)
        self.assertEqual(logged_output, {"boundary_indices": [15]})


class ExtraKeypointsIntegrationTests(unittest.TestCase):
    def test_vlm_method_writes_drop_in_goal_key(self):
        script_path = (
            Path(__file__).parents[1]
            / "external/mimicgen/mimicgen/scripts/generate_extra_keypoints.py"
        )
        spec = importlib.util.spec_from_file_location(
            "generate_extra_keypoints_test", script_path
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "demo_0"
            output = Path(directory) / "extra" / "demo_0"
            source.mkdir()
            for index in range(4):
                np.savez(
                    source / f"{index}.npz",
                    eef_pos=np.zeros((1, 3)),
                    eef_quat=np.array([[0.0, 0.0, 0.0, 1.0]]),
                    gripper_qpos=np.zeros((1, 2)),
                    action=np.zeros((1, 10)),
                    gripper_pcd=np.full((1, 4, 3), index, dtype=np.float32),
                    rgb_agentview=np.zeros((1, 8, 8, 3), dtype=np.uint8),
                )
            options = SimpleNamespace(
                mix_groups=None,
                vlm_camera="agentview",
                vlm_logs_dir="logs/subtask_boundaries",
            )
            with patch.object(module, "_compute_vlm_boundaries", return_value=[2]):
                length, record = module.process_demo(
                    str(source), str(output), ["vlm"], options, dump_indices=False
                )

            self.assertEqual(length, 4)
            # T-1 is required only by the generator's goal schedule.
            self.assertEqual(record["vlm"], [2, 3])
            with np.load(output / "0.npz") as data:
                self.assertEqual(data["goal_gripper_pcd_vlm"].shape, (1, 4, 3))


if __name__ == "__main__":
    unittest.main()
