"""Tests for PaddleOCR text detection normalization."""

from __future__ import annotations

import unittest

from text_redaction_benchmark.paddleocr_detector import (
    PaddleOCRTextDetector,
    PaddleOCRTextDetectorConfig,
    normalize_paddleocr_results,
    polygon_to_bbox_xyxy,
)


class FakePaddleOCRModel:
    def __init__(self, raw_results: list[dict]) -> None:
        """Store fake PaddleOCR results for later prediction calls."""
        self.raw_results = raw_results
        self.calls: list[tuple[str, int]] = []

    def predict(self, image: str, *, batch_size: int) -> list[dict]:
        """Return fake prediction results while recording call arguments."""
        self.calls.append((image, batch_size))
        return self.raw_results


class PaddleOCRDetectorTest(unittest.TestCase):
    def test_normalizes_documented_dt_polys_and_scores(self) -> None:
        """Check normalization of documented PaddleOCR dt_polys and dt_scores."""
        regions = normalize_paddleocr_results(
            [
                {
                    "res": {
                        "input_path": "sample.png",
                        "page_index": None,
                        "dt_polys": [
                            [[10, 20], [30, 20], [30, 40], [10, 40]],
                            [[100, 50], [150, 55], [145, 80], [95, 75]],
                        ],
                        "dt_scores": [0.91, 0.82],
                    }
                }
            ],
            source="PP-OCRv5_server_det",
        )

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            regions[0].polygon, ((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0))
        )
        self.assertEqual(regions[0].bbox_xyxy, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(regions[0].confidence, 0.91)
        self.assertEqual(regions[0].image_path, "sample.png")
        self.assertEqual(regions[0].detector, "paddleocr_text_detection")
        self.assertEqual(regions[1].bbox_xyxy, (95.0, 50.0, 150.0, 80.0))

    def test_detector_uses_injected_model_without_importing_paddleocr(self) -> None:
        """Check adapter prediction through an injected PaddleOCR-like model."""
        raw_results = [
            {
                "res": {
                    "input_path": "sample.png",
                    "page_index": None,
                    "dt_polys": [[[0, 0], [5, 0], [5, 10], [0, 10]]],
                    "dt_scores": [0.7],
                }
            }
        ]
        model = FakePaddleOCRModel(raw_results)
        detector = PaddleOCRTextDetector(
            PaddleOCRTextDetectorConfig(model_name="PP-OCRv5_server_det", batch_size=2),
            model=model,
        )

        regions = detector.predict("sample.png")

        self.assertEqual(model.calls, [("sample.png", 2)])
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].source, "PP-OCRv5_server_det")

    def test_rejects_mismatched_polygons_and_scores(self) -> None:
        """Check that malformed PaddleOCR output fails before normalization."""
        with self.assertRaises(ValueError):
            normalize_paddleocr_results(
                [
                    {
                        "res": {
                            "dt_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                            "dt_scores": [],
                        }
                    }
                ]
            )

    def test_polygon_to_bbox_rejects_empty_polygon(self) -> None:
        """Check that bbox derivation rejects empty polygons."""
        with self.assertRaises(ValueError):
            polygon_to_bbox_xyxy(())


if __name__ == "__main__":
    unittest.main()
