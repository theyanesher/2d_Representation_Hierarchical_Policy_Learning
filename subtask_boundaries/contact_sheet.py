"""Contact sheets built directly from numpy RGB trajectory frames."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_BOX_RESAMPLE = getattr(Image, "Resampling", Image).BOX


@dataclass(frozen=True)
class ContactSheet:
    """One contact-sheet image and the original indices visible on it."""

    image: Image.Image
    frame_indices: tuple[int, ...]

    def to_jpeg_bytes(self, quality: int = 95) -> bytes:
        output = io.BytesIO()
        self.image.save(output, format="JPEG", quality=quality, subsampling=2)
        return output.getvalue()

    def to_data_url(self, quality: int = 95) -> str:
        encoded = base64.b64encode(self.to_jpeg_bytes(quality=quality)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def normalize_rgb_frames(frames: np.ndarray) -> np.ndarray:
    """Validate RGB layout and return channel-last uint8 frames.

    Accepted layouts are ``(T, H, W, 3)`` and ``(T, 3, H, W)``. Floating
    inputs wholly inside ``[0, 1]`` are scaled to ``[0, 255]``; other numeric
    inputs are clipped to that byte range. NaN and infinity are made finite
    before conversion.
    """
    array = np.asarray(frames)
    if array.ndim != 4:
        raise ValueError(
            "RGB frames must be a 4D array shaped (T, H, W, 3) or (T, 3, H, W); "
            f"got shape {array.shape}"
        )
    if array.shape[0] == 0 or any(size == 0 for size in array.shape[1:]):
        raise ValueError(
            f"RGB frames must have non-empty dimensions; got shape {array.shape}"
        )

    if array.shape[-1] == 3:
        channel_last = array
    elif array.shape[1] == 3:
        channel_last = np.moveaxis(array, 1, -1)
    else:
        raise ValueError(
            "RGB frames must have exactly 3 channels in the last or second axis; "
            f"got shape {array.shape}"
        )

    if not (
        np.issubdtype(channel_last.dtype, np.number)
        or channel_last.dtype == np.bool_
    ):
        raise TypeError(f"RGB frames must have a numeric dtype; got {channel_last.dtype}")
    if channel_last.dtype == np.uint8:
        return np.ascontiguousarray(channel_last)

    values = np.nan_to_num(
        channel_last.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0
    )
    if np.issubdtype(channel_last.dtype, np.floating):
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum >= 0.0 and maximum <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(np.rint(values), 0, 255).astype(np.uint8))


def load_rgb_npz(path: str | Path, rgb_key: str = "rgb") -> np.ndarray:
    """Load and normalize one episode-level RGB array from an NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        if rgb_key not in data:
            raise KeyError(
                f"RGB key {rgb_key!r} not found in {path}; available keys: {data.files}"
            )
        return normalize_rgb_frames(data[rgb_key])


def sample_frame_indices(num_frames: int, sample_every_n_frames: int) -> list[int]:
    """Sample original indices and always include the terminal frame."""
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if sample_every_n_frames <= 0:
        raise ValueError("sample_every_n_frames must be > 0")
    indices = list(range(0, num_frames, sample_every_n_frames))
    terminal_index = num_frames - 1
    if indices[-1] != terminal_index:
        indices.append(terminal_index)
    return indices


def refinement_frame_indices(
    num_frames: int,
    coarse_index: int,
    radius: int,
    stride: int = 1,
) -> list[int]:
    """Build an inclusive, episode-clamped local window in original indices."""
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    coarse_index = min(max(int(coarse_index), 0), num_frames - 1)
    start = max(0, coarse_index - radius)
    stop = min(num_frames - 1, coarse_index + radius)
    indices = list(range(start, stop + 1, stride))
    if coarse_index not in indices:
        indices.append(coarse_index)
        indices.sort()
    return indices


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, 17)
        except OSError:
            pass
    return ImageFont.load_default()


def _frame_label(frame_index: int) -> str:
    return f"FRAME {frame_index}"


def _draw_frame_badge(image: Image.Image, frame_index: int) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = _load_font()
    label = _frame_label(frame_index)
    try:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width, height = right - left, bottom - top
    except AttributeError:  # Pillow < 8 compatibility
        width, height = draw.textsize(label, font=font)
    draw.rectangle((0, 0, width + 14, height + 10), fill=(0, 0, 0))
    draw.text((7, 4), label, fill=(255, 255, 255), font=font)
    return result


def make_contact_sheets(
    frames: np.ndarray,
    frame_indices: Sequence[int],
    *,
    frame_width: int = 224,
    frames_per_sheet: int = 20,
    columns: int = 5,
    sheet_overlap_frames: int = 2,
) -> list[ContactSheet]:
    """Render original-index-badged RGB frames into overlapping contact sheets.

    Adjacent sheets repeat ``sheet_overlap_frames`` indices so a transition at
    a sheet seam retains temporal context. Partial final sheets allocate only
    the rows needed by their populated tiles.
    """
    rgb = normalize_rgb_frames(frames)
    if frame_width <= 0:
        raise ValueError("frame_width must be > 0")
    if frames_per_sheet <= 0:
        raise ValueError("frames_per_sheet must be > 0")
    if columns <= 0:
        raise ValueError("columns must be > 0")
    if columns > frames_per_sheet:
        raise ValueError("columns must be <= frames_per_sheet")
    if sheet_overlap_frames < 0 or sheet_overlap_frames >= frames_per_sheet:
        raise ValueError(
            "sheet_overlap_frames must satisfy "
            "0 <= sheet_overlap_frames < frames_per_sheet"
        )

    indices = [int(index) for index in frame_indices]
    invalid = [index for index in indices if index < 0 or index >= len(rgb)]
    if invalid:
        raise IndexError(
            f"frame_indices must lie in [0, {len(rgb) - 1}]; invalid values: {invalid}"
        )
    if not indices:
        return []

    tile_height = max(1, round(rgb.shape[1] * frame_width / rgb.shape[2]))
    sheets: list[ContactSheet] = []
    offset = 0
    step = frames_per_sheet - sheet_overlap_frames
    while offset < len(indices):
        chunk = indices[offset : offset + frames_per_sheet]
        rows = math.ceil(len(chunk) / columns)
        sheet = Image.new(
            "RGB", (frame_width * columns, tile_height * rows), color=(0, 0, 0)
        )
        for tile_index, original_index in enumerate(chunk):
            tile = Image.fromarray(rgb[original_index], mode="RGB")
            if tile.width != frame_width:
                tile = tile.resize((frame_width, tile_height), resample=_BOX_RESAMPLE)
            tile = _draw_frame_badge(tile, original_index)
            x = (tile_index % columns) * frame_width
            y = (tile_index // columns) * tile_height
            sheet.paste(tile, (x, y))
        sheets.append(ContactSheet(image=sheet, frame_indices=tuple(chunk)))
        if offset + frames_per_sheet >= len(indices):
            break
        offset += step
    return sheets
