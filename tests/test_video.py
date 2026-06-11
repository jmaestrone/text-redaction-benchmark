"""Tests for offline video redaction orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from text_redaction_benchmark.schemas import TextRegion
from text_redaction_benchmark.video import VideoRedactionConfig, redact_video_offline


class FakeVideoCapture:
    """Small cv2.VideoCapture replacement for deterministic frame reads."""

    def __init__(self, frames: list[np.ndarray], *, fps: float) -> None:
        """Store frames and capture metadata."""
        self.frames = frames
        self.fps = fps
        self.read_index = 0
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake capture opened successfully."""
        return True

    def get(self, property_id: int) -> float:
        """Return fake OpenCV capture properties."""
        if property_id == FakeCv2.CAP_PROP_FPS:
            return self.fps
        if property_id == FakeCv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1])
        if property_id == FakeCv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0])
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return frames until the fake capture is exhausted."""
        if self.read_index >= len(self.frames):
            return (False, None)
        frame = self.frames[self.read_index]
        self.read_index += 1
        return (True, frame.copy())

    def release(self) -> None:
        """Mark the fake capture as released."""
        self.released = True


class FakeVideoWriter:
    """Small cv2.VideoWriter replacement that stores written frames."""

    def __init__(
        self,
        output_path: str,
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        """Store writer metadata and written frames."""
        self.output_path = output_path
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        """Return whether the fake writer opened successfully."""
        return True

    def write(self, frame: np.ndarray) -> None:
        """Store a written video frame."""
        self.frames.append(frame.copy())

    def release(self) -> None:
        """Mark the fake writer as released."""
        self.released = True


class FakeCv2:
    """Small subset of cv2 needed by the video redaction workflow."""

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, frames: list[np.ndarray], *, fps: float = 24.0) -> None:
        """Store fake input frames and expose the last created writer."""
        self.frames = frames
        self.fps = fps
        self.last_capture: FakeVideoCapture | None = None
        self.last_writer: FakeVideoWriter | None = None

    def VideoCapture(self, input_path: str) -> FakeVideoCapture:
        """Create a fake video capture."""
        self.last_capture = FakeVideoCapture(self.frames, fps=self.fps)
        return self.last_capture

    def VideoWriter(
        self,
        output_path: str,
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> FakeVideoWriter:
        """Create a fake video writer."""
        self.last_writer = FakeVideoWriter(output_path, fourcc, fps, frame_size)
        return self.last_writer

    def VideoWriter_fourcc(self, *codec: str) -> int:
        """Return a deterministic fake fourcc value."""
        return len(codec)


class FakeDetector:
    """Detector replacement that returns one region for the first frame only."""

    def __init__(self) -> None:
        """Initialize recorded detector calls."""
        self.calls: list[tuple[Any, str | None, int | None]] = []

    def predict(
        self,
        image: Any,
        *,
        image_path: str | None = None,
        frame_index: int | None = None,
    ) -> list[TextRegion]:
        """Return a text region for frame zero and no regions afterwards."""
        self.calls.append((image, image_path, frame_index))
        if frame_index != 0:
            return []
        return [
            TextRegion(
                polygon=((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)),
                bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
                confidence=0.9,
                source="fake",
                detector="fake",
                image_path=image_path,
                frame_index=frame_index,
            )
        ]


class VideoRedactionTest(unittest.TestCase):
    def test_redact_video_offline_writes_artifacts_for_every_frame(self) -> None:
        """Check that video redaction writes JSONL, summaries, and frames."""
        first_frame = np.full((4, 4, 3), 255, dtype=np.uint8)
        second_frame = np.full((4, 4, 3), 255, dtype=np.uint8)
        fake_cv2 = FakeCv2([first_frame, second_frame], fps=12.0)
        fake_detector = FakeDetector()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = VideoRedactionConfig(
                input_video=Path("input-videos/sample.mp4"),
                output_root=Path(temp_dir),
                run_id="test-run",
                mask_color="black",
            )

            result = redact_video_offline(config, fake_detector, cv2_module=fake_cv2)

            self.assertEqual(result.frame_count, 2)
            self.assertEqual(result.text_region_count, 1)
            self.assertEqual(result.fps, 12.0)
            self.assertEqual(result.width, 4)
            self.assertEqual(result.height, 4)
            self.assertEqual([call[2] for call in fake_detector.calls], [0, 1])
            self.assertIsNotNone(fake_cv2.last_writer)
            self.assertEqual(len(fake_cv2.last_writer.frames), 2)
            self.assertEqual(tuple(fake_cv2.last_writer.frames[0][1, 1]), (0, 0, 0))

            jsonl_lines = result.paths.prediction_jsonl.read_text().splitlines()
            self.assertEqual(len(jsonl_lines), 2)
            first_prediction = json.loads(jsonl_lines[0])
            self.assertEqual(first_prediction["frame_index"], 0)
            self.assertEqual(len(first_prediction["regions"]), 1)

            summary = json.loads(result.paths.summary_json.read_text())
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["text_region_count"], 1)
            self.assertFalse(summary["audio_preserved"])
            self.assertTrue(result.paths.summary_csv.exists())


if __name__ == "__main__":
    unittest.main()
