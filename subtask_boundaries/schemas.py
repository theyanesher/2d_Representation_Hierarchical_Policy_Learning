"""Structured response schemas for boundary-only VLM predictions."""

from pydantic import BaseModel, ConfigDict


class BoundaryPrediction(BaseModel):
    """Coarse temporal boundaries, expressed as original frame indices."""

    model_config = ConfigDict(extra="forbid")
    boundary_indices: list[int]


class RefinedBoundary(BaseModel):
    """One refined temporal boundary, expressed as an original frame index."""

    model_config = ConfigDict(extra="forbid")
    boundary_index: int

