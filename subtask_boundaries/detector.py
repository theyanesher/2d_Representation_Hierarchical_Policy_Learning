"""Coarse-to-fine temporal boundary detection for numpy RGB trajectories."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol, TypeVar
from uuid import uuid4

import numpy as np
from pydantic import BaseModel

from .contact_sheet import (
    ContactSheet,
    load_rgb_npz,
    make_contact_sheets,
    normalize_rgb_frames,
    refinement_frame_indices,
    sample_frame_indices,
)
from .prompts import coarse_prompt, refinement_prompt
from .schemas import BoundaryPrediction, RefinedBoundary

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class BoundaryVLM(Protocol):
    """Small provider interface used by the detector and its tests."""

    def predict(
        self,
        *,
        prompt: str,
        contact_sheets: list[ContactSheet],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        """Return one response parsed into ``schema``."""


class OpenAIBoundaryVLM:
    """OpenAI Responses API adapter with Pydantic structured outputs."""

    def __init__(
        self,
        model: str = "gpt-5.4",
        *,
        client: object | None = None,
        image_detail: str = "high",
        jpeg_quality: int = 95,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.image_detail = image_detail
        self.jpeg_quality = jpeg_quality

    def predict(
        self,
        *,
        prompt: str,
        contact_sheets: list[ContactSheet],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        if not contact_sheets:
            raise ValueError("at least one contact sheet is required")
        content = [{"type": "input_text", "text": prompt}]
        content.extend(
            {
                "type": "input_image",
                "image_url": sheet.to_data_url(quality=self.jpeg_quality),
                "detail": self.image_detail,
            }
            for sheet in contact_sheets
        )
        response = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("VLM returned no parsed structured output")
        return parsed


class GeminiBoundaryVLM:
    """Google Gemini Interactions API adapter with Pydantic validation."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        *,
        client: object | None = None,
        jpeg_quality: int = 95,
    ) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "Gemini support requires google-genai; install the repository "
                    "environment with `pixi install`"
                ) from exc

            client = genai.Client()
        self.client = client
        self.model = model
        self.jpeg_quality = jpeg_quality

    def predict(
        self,
        *,
        prompt: str,
        contact_sheets: list[ContactSheet],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        if not contact_sheets:
            raise ValueError("at least one contact sheet is required")
        interaction_input = [{"type": "text", "text": prompt}]
        interaction_input.extend(
            {
                "type": "image",
                "data": base64.b64encode(
                    sheet.to_jpeg_bytes(quality=self.jpeg_quality)
                ).decode("ascii"),
                "mime_type": "image/jpeg",
            }
            for sheet in contact_sheets
        )
        interaction = self.client.interactions.create(
            model=self.model,
            input=interaction_input,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
        )
        output_text = interaction.output_text
        if not output_text:
            raise RuntimeError("Gemini returned no structured output text")
        return schema.model_validate_json(output_text)


class QwenBoundaryVLM:
    """Local Qwen vLLM adapter with Pydantic structured output.

    The local server is expected to expose an OpenAI-compatible endpoint. The
    repository's ``scripts/serve_qwen_local.sh`` serves the model name used by
    default, so no API key is required.
    """

    def __init__(
        self,
        model: str = "qwen3.6-local",
        *,
        client: object | None = None,
        base_url: str | None = None,
        jpeg_quality: int = 95,
    ) -> None:
        if client is None:
            from openai import OpenAI

            base_url = base_url or os.getenv(
                "QWEN_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"
            )
            client = OpenAI(api_key="local", base_url=base_url)
        self.client = client
        self.model = model
        self.jpeg_quality = jpeg_quality

    def predict(
        self,
        *,
        prompt: str,
        contact_sheets: list[ContactSheet],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        if not contact_sheets:
            raise ValueError("at least one contact sheet is required")
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": sheet.to_data_url(quality=self.jpeg_quality),
                },
            }
            for sheet in contact_sheets
        ]
        content.append({"type": "text", "text": prompt})
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format=schema,
            max_completion_tokens=256,
            temperature=0,
            # Qwen's local API uses chat-template kwargs for non-thinking mode.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("local Qwen returned no parsed structured output")
        return parsed


class QwenCloudBoundaryVLM:
    """Optional QwenCloud adapter retained for hosted deployments."""

    def __init__(
        self,
        model: str = "qwen3.6-flash",
        *,
        client: object | None = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        jpeg_quality: int = 95,
    ) -> None:
        if client is None:
            from openai import OpenAI

            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
            if not api_key:
                raise ValueError(
                    "QwenCloud requires DASHSCOPE_API_KEY (or QWEN_API_KEY)"
                )
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.jpeg_quality = jpeg_quality

    def predict(
        self,
        *,
        prompt: str,
        contact_sheets: list[ContactSheet],
        schema: type[StructuredResult],
    ) -> StructuredResult:
        if not contact_sheets:
            raise ValueError("at least one contact sheet is required")
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": sheet.to_data_url(quality=self.jpeg_quality),
                },
            }
            for sheet in contact_sheets
        ]
        content.append({"type": "text", "text": prompt})
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format=schema,
            extra_body={"enable_thinking": False},
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("QwenCloud returned no parsed structured output")
        return parsed


def sanitize_boundary_indices(
    indices: list[int] | tuple[int, ...] | np.ndarray,
    num_frames: int,
    *,
    min_boundary_distance_frames: int = 0,
) -> list[int]:
    """Clamp, sort, deduplicate, and conservatively merge boundary indices.

    Out-of-range values are clamped to the nearest episode edge. When
    ``min_boundary_distance_frames`` is positive, sorted predictions closer
    than that distance are merged by retaining the earlier prediction. The
    default of zero performs no distance-based merging.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if min_boundary_distance_frames < 0:
        raise ValueError("min_boundary_distance_frames must be >= 0")
    clamped = sorted(
        {
            min(max(int(index), 0), num_frames - 1)
            for index in indices
        }
    )
    if min_boundary_distance_frames == 0:
        return clamped
    kept: list[int] = []
    for index in clamped:
        if not kept or index - kept[-1] >= min_boundary_distance_frames:
            kept.append(index)
    return kept


def detect_coarse_boundaries(
    frames: np.ndarray,
    *,
    vlm: BoundaryVLM,
    sample_every_n_frames: int = 15,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    sheet_overlap_frames: int = 2,
    instruction: str | None = None,
    logs_dir: str | Path | None = None,
) -> list[int]:
    """Predict sparse boundaries, restricted to visibly sampled indices."""
    rgb = normalize_rgb_frames(frames)
    sampled = sample_frame_indices(len(rgb), sample_every_n_frames)
    sheets = make_contact_sheets(
        rgb,
        sampled,
        frame_width=frame_width,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        sheet_overlap_frames=sheet_overlap_frames,
    )
    prompt = coarse_prompt(instruction)
    coarse_log_dir = Path(logs_dir) if logs_dir is not None else None
    if coarse_log_dir is not None:
        coarse_log_dir.mkdir(parents=True, exist_ok=True)
        (coarse_log_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (coarse_log_dir / "input.json").write_text(
            json.dumps(
                {
                    "num_trajectory_frames": len(rgb),
                    "sample_every_n_frames": sample_every_n_frames,
                    "sheet_overlap_frames": sheet_overlap_frames,
                    "sampled_original_frame_indices": sampled,
                    "contact_sheets": [
                        {
                            "file": f"contact_sheet_{index:03d}.jpg",
                            "original_frame_indices": list(sheet.frame_indices),
                        }
                        for index, sheet in enumerate(sheets, start=1)
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for index, sheet in enumerate(sheets, start=1):
            (coarse_log_dir / f"contact_sheet_{index:03d}.jpg").write_bytes(
                sheet.to_jpeg_bytes()
            )
    try:
        result = vlm.predict(
            prompt=prompt,
            contact_sheets=sheets,
            schema=BoundaryPrediction,
        )
    except Exception as exc:
        if coarse_log_dir is not None:
            (coarse_log_dir / "error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
        raise
    if coarse_log_dir is not None:
        (coarse_log_dir / "output.json").write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    sampled_set = set(sampled)
    return sorted(
        {int(index) for index in result.boundary_indices if int(index) in sampled_set}
    )


def refine_boundary(
    frames: np.ndarray,
    coarse_index: int,
    *,
    vlm: BoundaryVLM,
    refinement_radius: int = 15,
    refinement_stride: int = 1,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    sheet_overlap_frames: int = 2,
    instruction: str | None = None,
) -> int:
    """Refine one coarse prediction using an inclusive dense local window."""
    rgb = normalize_rgb_frames(frames)
    local_indices = refinement_frame_indices(
        len(rgb), coarse_index, refinement_radius, refinement_stride
    )
    sheets = make_contact_sheets(
        rgb,
        local_indices,
        frame_width=frame_width,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        sheet_overlap_frames=sheet_overlap_frames,
    )
    result = vlm.predict(
        prompt=refinement_prompt(coarse_index, instruction),
        contact_sheets=sheets,
        schema=RefinedBoundary,
    )
    predicted = int(result.boundary_index)
    if predicted not in set(local_indices):
        # Keep the provider contract strict: refinement may only select a
        # frame it actually saw. The coarse index is guaranteed to be shown.
        return min(local_indices, key=lambda index: (abs(index - predicted), index))
    return predicted


def detect_subtask_boundaries(
    frames: np.ndarray,
    *,
    vlm: BoundaryVLM | None = None,
    provider: Literal["qwen", "qwen_cloud", "openai", "gemini"] = "qwen",
    model: str | None = None,
    qwen_base_url: str | None = None,
    sample_every_n_frames: int = 15,
    refine: bool = True,
    stop_after_sparse_annotation: bool = False,
    refinement_radius: int = 15,
    refinement_stride: int = 1,
    min_boundary_distance_frames: int = 0,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    sheet_overlap_frames: int = 2,
    instruction: str | None = None,
    logs_dir: str | Path | None = "logs/subtask_boundaries",
) -> list[int]:
    """Return sorted, unique temporal transitions in original frame indices.

    The initial sparse request and response are logged to a unique run folder
    below ``logs_dir`` by default. Pass ``logs_dir=None`` to opt out.
    """
    rgb = normalize_rgb_frames(frames)
    if provider not in {"qwen", "qwen_cloud", "openai", "gemini"}:
        raise ValueError(
            "provider must be 'qwen', 'qwen_cloud', 'openai', or 'gemini'"
        )
    if vlm is not None:
        backend = vlm
    elif provider == "gemini":
        backend = GeminiBoundaryVLM(model=model or "gemini-3.5-flash")
    elif provider == "qwen":
        backend = QwenBoundaryVLM(
            model=model or "qwen3.6-local", base_url=qwen_base_url
        )
    elif provider == "qwen_cloud":
        backend = QwenCloudBoundaryVLM(model=model or "qwen3.6-flash")
    else:
        backend = OpenAIBoundaryVLM(model=model or "gpt-5.4")
    run_log_dir = None
    if logs_dir is not None:
        run_id = "run_{}_{}_{}".format(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
            os.getpid(),
            uuid4().hex[:8],
        )
        run_log_dir = Path(logs_dir) / run_id
    coarse = detect_coarse_boundaries(
        rgb,
        vlm=backend,
        sample_every_n_frames=sample_every_n_frames,
        frame_width=frame_width,
        frames_per_sheet=frames_per_sheet,
        columns=columns,
        sheet_overlap_frames=sheet_overlap_frames,
        instruction=instruction,
        logs_dir=run_log_dir,
    )
    predictions = coarse
    if refine and not stop_after_sparse_annotation:
        predictions = [
            refine_boundary(
                rgb,
                index,
                vlm=backend,
                refinement_radius=refinement_radius,
                refinement_stride=refinement_stride,
                frame_width=frame_width,
                frames_per_sheet=frames_per_sheet,
                columns=columns,
                sheet_overlap_frames=sheet_overlap_frames,
                instruction=instruction,
            )
            for index in coarse
        ]
    return sanitize_boundary_indices(
        predictions,
        len(rgb),
        min_boundary_distance_frames=min_boundary_distance_frames,
    )


def detect_subtask_boundaries_npz(
    path: str | Path,
    rgb_key: str = "rgb",
    **kwargs: object,
) -> list[int]:
    """Load an episode-level NPZ RGB array and detect its transitions."""
    return detect_subtask_boundaries(load_rgb_npz(path, rgb_key), **kwargs)
