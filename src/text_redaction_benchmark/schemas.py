"""Shared data schemas for text detection and redaction workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]
Polygon = tuple[Point, ...]
BBoxXYXY = tuple[float, float, float, float]


@dataclass
class TextRegion:
    """Normalized text detection region emitted by detector adapters."""

    polygon: Polygon
    bbox_xyxy: BBoxXYXY
    confidence: float
    source: str
    detector: str
    image_path: str | None = None
    frame_index: int | None = None
    page_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this text region."""
        return {
            "polygon": [[x, y] for x, y in self.polygon],
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
            "source": self.source,
            "detector": self.detector,
            "image_path": self.image_path,
            "frame_index": self.frame_index,
            "page_index": self.page_index,
            "metadata": self.metadata,
        }


@dataclass
class FramePredictionRecord:
    """One JSONL-friendly prediction record for a single image or video frame."""

    image_path: str | None
    frame_index: int | None
    width: int | None
    height: int | None
    regions: list[TextRegion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this frame record."""
        return {
            "image_path": self.image_path,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "regions": [region.to_dict() for region in self.regions],
            "metadata": self.metadata,
        }
