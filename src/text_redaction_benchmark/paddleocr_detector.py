"""PaddleOCR text detection adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from text_redaction_benchmark.schemas import BBoxXYXY, Point, Polygon, TextRegion

DEFAULT_PADDLEOCR_TEXT_DETECTION_MODEL = "PP-OCRv5_server_det"
DETECTOR_NAME = "paddleocr_text_detection"


@dataclass(frozen=True)
class PaddleOCRTextDetectorConfig:
    """Configuration for PaddleOCR text detection."""

    model_name: str = DEFAULT_PADDLEOCR_TEXT_DETECTION_MODEL
    model_dir: str | None = None
    device: str | None = None
    thresh: float | None = None
    box_thresh: float | None = None
    unclip_ratio: float | None = None
    limit_side_len: int | None = None
    limit_type: str | None = None
    batch_size: int = 1

    def model_kwargs(self) -> dict[str, Any]:
        """Return non-empty keyword arguments for PaddleOCR TextDetection."""
        config_values = {
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "device": self.device,
            "thresh": self.thresh,
            "box_thresh": self.box_thresh,
            "unclip_ratio": self.unclip_ratio,
            "limit_side_len": self.limit_side_len,
            "limit_type": self.limit_type,
        }
        return {
            config_key: config_value
            for config_key, config_value in config_values.items()
            if config_value is not None
        }


class PaddleOCRTextDetector:
    """Detect text regions with PaddleOCR's TextDetection module."""

    def __init__(
        self,
        config: PaddleOCRTextDetectorConfig | None = None,
        model: Any | None = None,
    ) -> None:
        """Create a detector, optionally using an injected PaddleOCR-like model."""
        self.config = config or PaddleOCRTextDetectorConfig()
        self._model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        """Load PaddleOCR's TextDetection model lazily."""
        try:
            from paddleocr import TextDetection
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install this project with the "
                "`paddleocr` extra before running PaddleOCR-backed detection."
            ) from exc
        return TextDetection(**self.config.model_kwargs())

    def predict(
        self,
        image: Any,
        *,
        image_path: str | None = None,
        frame_index: int | None = None,
    ) -> list[TextRegion]:
        """Run PaddleOCR text detection and return normalized text regions."""
        raw_results = self._model.predict(image, batch_size=self.config.batch_size)
        return normalize_paddleocr_results(
            raw_results,
            detector=DETECTOR_NAME,
            source=self.config.model_name,
            image_path=image_path,
            frame_index=frame_index,
        )


def _extract_result_payload(raw_result: Any) -> Mapping[str, Any]:
    """Extract the result mapping from PaddleOCR's supported result wrappers."""
    if isinstance(raw_result, Mapping):
        result_payload = raw_result.get("res", raw_result)
    elif isinstance(getattr(raw_result, "json", None), Mapping):
        result_payload = raw_result.json.get("res", raw_result.json)
    elif isinstance(getattr(raw_result, "res", None), Mapping):
        result_payload = raw_result.res
    else:
        raise TypeError("Unsupported PaddleOCR result shape.")

    if not isinstance(result_payload, Mapping):
        raise TypeError("PaddleOCR result payload must be a mapping.")
    return result_payload


def _normalize_polygon(raw_polygon: Any) -> Polygon:
    """Convert a PaddleOCR polygon-like value to normalized point tuples."""
    points = _as_sequence(raw_polygon)
    if len(points) < 3:
        raise ValueError("Text region polygons must contain at least three points.")
    return tuple(_normalize_point(point) for point in points)


def _normalize_point(raw_point: Any) -> Point:
    """Convert a PaddleOCR point-like value to an `(x, y)` tuple."""
    coordinates = _as_sequence(raw_point)
    if len(coordinates) != 2:
        raise ValueError(
            "Text region polygon points must have exactly two coordinates."
        )
    return (float(coordinates[0]), float(coordinates[1]))


def _as_sequence(value: Any) -> Sequence[Any]:
    """Return list-like values as sequences, including numpy-style arrays."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError("Expected a sequence-like value.")


def polygon_to_bbox_xyxy(polygon: Polygon) -> BBoxXYXY:
    """Return the axis-aligned bbox enclosing a polygon."""
    if not polygon:
        raise ValueError("Cannot derive a bounding box from an empty polygon.")
    x_coordinates = [point[0] for point in polygon]
    y_coordinates = [point[1] for point in polygon]
    return (
        min(x_coordinates),
        min(y_coordinates),
        max(x_coordinates),
        max(y_coordinates),
    )


def normalize_paddleocr_results(
    raw_results: Iterable[Any],
    *,
    detector: str = DETECTOR_NAME,
    source: str = DEFAULT_PADDLEOCR_TEXT_DETECTION_MODEL,
    image_path: str | None = None,
    frame_index: int | None = None,
) -> list[TextRegion]:
    """Normalize PaddleOCR `dt_polys` and `dt_scores` into text regions."""
    text_regions: list[TextRegion] = []
    for result_index, raw_result in enumerate(raw_results):
        result_payload = _extract_result_payload(raw_result)
        polygons = _as_sequence(result_payload.get("dt_polys", ()))
        scores = _as_sequence(result_payload.get("dt_scores", ()))
        if len(polygons) != len(scores):
            raise ValueError(
                "PaddleOCR result contains mismatched `dt_polys` and `dt_scores` lengths."
            )

        result_image_path = image_path or result_payload.get("input_path")
        page_index = result_payload.get("page_index")
        for region_index, raw_polygon in enumerate(polygons):
            polygon = _normalize_polygon(raw_polygon)
            text_regions.append(
                TextRegion(
                    polygon=polygon,
                    bbox_xyxy=polygon_to_bbox_xyxy(polygon),
                    confidence=float(scores[region_index]),
                    source=source,
                    detector=detector,
                    image_path=result_image_path,
                    frame_index=frame_index,
                    page_index=page_index,
                    metadata={
                        "result_index": result_index,
                        "region_index": region_index,
                    },
                )
            )
    return text_regions
