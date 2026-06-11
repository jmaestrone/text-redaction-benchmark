"""Tests for RF-DETR text baseline normalization."""

from __future__ import annotations

import unittest

from text_redaction_benchmark.rfdetr_detector import (
    RFDETRTextDetector,
    RFDETRTextDetectorConfig,
    normalize_rfdetr_detections,
)


class FakeDetections:
    """Small Supervision-like detections object for RF-DETR tests."""

    def __init__(
        self,
        *,
        xyxy: list[list[float]],
        confidence: list[float],
        class_id: list[int],
    ) -> None:
        """Store detection fields with RF-DETR/Supervision-style names."""
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id


class FakeRFDETRModel:
    """Small RF-DETR-like model for injected adapter tests."""

    def __init__(self, detections: FakeDetections) -> None:
        """Store fake detections for later prediction calls."""
        self.detections = detections
        self.calls: list[tuple[str, float]] = []

    def predict(self, image: str, *, threshold: float) -> FakeDetections:
        """Return fake detections while recording call arguments."""
        self.calls.append((image, threshold))
        return self.detections


class RFDETRDetectorTest(unittest.TestCase):
    def test_normalizes_boxes_to_rectangular_text_regions(self) -> None:
        """Check that RF-DETR xyxy boxes become rectangular text polygons."""
        detections = FakeDetections(
            xyxy=[[10.0, 20.0, 30.0, 40.0]],
            confidence=[0.87],
            class_id=[0],
        )

        regions = normalize_rfdetr_detections(
            detections,
            source="rfdetr-medium",
            image_path="sample.png",
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox_xyxy, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(
            regions[0].polygon,
            ((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)),
        )
        self.assertEqual(regions[0].confidence, 0.87)
        self.assertEqual(regions[0].detector, "rfdetr_text_baseline")
        self.assertEqual(regions[0].image_path, "sample.png")

    def test_filters_non_text_class_ids_by_default(self) -> None:
        """Check that the default one-class text id keeps only class zero."""
        detections = {
            "xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "confidence": [0.9, 0.8],
            "class_id": [0, 3],
        }

        regions = normalize_rfdetr_detections(detections)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].metadata["class_id"], 0)

    def test_can_keep_all_class_ids_for_exploration(self) -> None:
        """Check that class filtering can be disabled for exploratory baselines."""
        detections = {
            "xyxy": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "confidence": [0.9, 0.8],
            "class_id": [0, 3],
        }

        regions = normalize_rfdetr_detections(detections, text_class_id=None)

        self.assertEqual(len(regions), 2)

    def test_detector_uses_injected_model_without_importing_rfdetr(self) -> None:
        """Check adapter prediction through an injected RF-DETR-like model."""
        detections = FakeDetections(
            xyxy=[[1.0, 2.0, 3.0, 4.0]],
            confidence=[0.75],
            class_id=[0],
        )
        model = FakeRFDETRModel(detections)
        detector = RFDETRTextDetector(
            RFDETRTextDetectorConfig(model_size="small", threshold=0.42),
            model=model,
        )

        regions = detector.predict("sample.png", frame_index=5)

        self.assertEqual(model.calls, [("sample.png", 0.42)])
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].source, "rfdetr-small")
        self.assertEqual(regions[0].frame_index, 5)

    def test_rejects_mismatched_detection_lengths(self) -> None:
        """Check that malformed RF-DETR field lengths fail before conversion."""
        detections = {"xyxy": [[0, 0, 1, 1]], "confidence": [], "class_id": [0]}

        with self.assertRaises(ValueError):
            normalize_rfdetr_detections(detections)

    def test_rejects_invalid_xyxy_order(self) -> None:
        """Check that boxes must use valid xyxy coordinate order."""
        detections = {"xyxy": [[10, 0, 1, 1]], "confidence": [0.9], "class_id": [0]}

        with self.assertRaises(ValueError):
            normalize_rfdetr_detections(detections)


if __name__ == "__main__":
    unittest.main()
