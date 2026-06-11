"""RF-DETR box-based text detection baseline adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from text_redaction_benchmark.schemas import BBoxXYXY, Polygon, TextRegion

RFDETRModelSize = Literal["nano", "small", "medium", "large"]

DEFAULT_RFDETR_MODEL_SIZE: RFDETRModelSize = "medium"
DEFAULT_RFDETR_TEXT_CLASS_ID = 0
DETECTOR_NAME = "rfdetr_text_baseline"


@dataclass(frozen=True)
class RFDETRTextDetectorConfig:
    """Configuration for an RF-DETR text baseline detector."""

    model_size: RFDETRModelSize = DEFAULT_RFDETR_MODEL_SIZE
    threshold: float = 0.5
    text_class_id: int | None = DEFAULT_RFDETR_TEXT_CLASS_ID
    model_kwargs: Mapping[str, Any] | None = None


class RFDETRTextDetector:
    """Detect text boxes with an RF-DETR baseline model."""

    def __init__(
        self,
        config: RFDETRTextDetectorConfig | None = None,
        model: Any | None = None,
    ) -> None:
        """Create a detector, optionally using an injected RF-DETR-like model."""
        self.config = config or RFDETRTextDetectorConfig()
        self._model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        """Load the configured RF-DETR model lazily."""
        model_class = _load_rfdetr_model_class(self.config.model_size)
        model_kwargs = dict(self.config.model_kwargs or {})
        return model_class(**model_kwargs)

    def predict(
        self,
        image: Any,
        *,
        image_path: str | None = None,
        frame_index: int | None = None,
    ) -> list[TextRegion]:
        """Run RF-DETR prediction and return normalized text regions."""
        detections = self._model.predict(image, threshold=self.config.threshold)
        return normalize_rfdetr_detections(
            detections,
            source=f"rfdetr-{self.config.model_size}",
            text_class_id=self.config.text_class_id,
            image_path=image_path,
            frame_index=frame_index,
        )


def _load_rfdetr_model_class(model_size: RFDETRModelSize) -> type[Any]:
    """Import the RF-DETR model class for a supported model size."""
    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except ImportError as exc:
        raise RuntimeError(
            "RF-DETR is not installed. Install the `rfdetr` package before "
            "running RF-DETR-backed detection."
        ) from exc

    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    return model_classes[model_size]


def _as_sequence(value: Any) -> Sequence[Any]:
    """Return list-like values as sequences, including numpy-style arrays."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError("Expected a sequence-like value.")


def _extract_detection_field(detections: Any, field_name: str) -> Sequence[Any]:
    """Extract a field from a Supervision-like detections object or mapping."""
    if isinstance(detections, Mapping):
        if field_name not in detections:
            raise ValueError(f"RF-DETR detections missing `{field_name}`.")
        field_value = detections[field_name]
    else:
        field_value = getattr(detections, field_name, None)
        if field_value is None:
            raise ValueError(f"RF-DETR detections missing `{field_name}`.")
    return _as_sequence(field_value)


def _normalize_bbox_xyxy(raw_bbox: Any) -> BBoxXYXY:
    """Convert an RF-DETR bbox-like value into `(x1, y1, x2, y2)`."""
    coordinates = _as_sequence(raw_bbox)
    if len(coordinates) != 4:
        raise ValueError("RF-DETR boxes must contain exactly four coordinates.")
    x_min, y_min, x_max, y_max = (float(coordinate) for coordinate in coordinates)
    if x_max < x_min or y_max < y_min:
        raise ValueError("RF-DETR boxes must be in xyxy order.")
    return (x_min, y_min, x_max, y_max)


def _bbox_to_polygon(bbox_xyxy: BBoxXYXY) -> Polygon:
    """Convert an axis-aligned bbox to a rectangular polygon."""
    x_min, y_min, x_max, y_max = bbox_xyxy
    return ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))


def _class_matches(class_id: int | None, text_class_id: int | None) -> bool:
    """Return whether a detection should be treated as text."""
    return text_class_id is None or class_id == text_class_id


def _validate_detection_lengths(
    boxes: Sequence[Any],
    confidences: Sequence[Any],
    class_ids: Sequence[Any],
) -> None:
    """Check that RF-DETR detection arrays have matching lengths."""
    if len(boxes) != len(confidences) or len(boxes) != len(class_ids):
        raise ValueError("RF-DETR detections contain mismatched field lengths.")


def normalize_rfdetr_detections(
    detections: Any,
    *,
    source: str = f"rfdetr-{DEFAULT_RFDETR_MODEL_SIZE}",
    text_class_id: int | None = DEFAULT_RFDETR_TEXT_CLASS_ID,
    detector: str = DETECTOR_NAME,
    image_path: str | None = None,
    frame_index: int | None = None,
) -> list[TextRegion]:
    """Normalize RF-DETR box detections into text regions."""
    boxes = _extract_detection_field(detections, "xyxy")
    confidences = _extract_detection_field(detections, "confidence")
    class_ids = _extract_detection_field(detections, "class_id")
    _validate_detection_lengths(boxes, confidences, class_ids)

    text_regions: list[TextRegion] = []
    for detection_index, raw_bbox in enumerate(boxes):
        class_id = int(class_ids[detection_index])
        if not _class_matches(class_id, text_class_id):
            continue

        bbox_xyxy = _normalize_bbox_xyxy(raw_bbox)
        text_regions.append(
            TextRegion(
                polygon=_bbox_to_polygon(bbox_xyxy),
                bbox_xyxy=bbox_xyxy,
                confidence=float(confidences[detection_index]),
                source=source,
                detector=detector,
                image_path=image_path,
                frame_index=frame_index,
                metadata={
                    "detection_index": detection_index,
                    "class_id": class_id,
                    "geometry_source": "bbox_xyxy",
                },
            )
        )
    return text_regions
