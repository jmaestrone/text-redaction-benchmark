"""Offline every-frame video text redaction workflows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from text_redaction_benchmark.redaction import MaskColor, apply_solid_redaction
from text_redaction_benchmark.schemas import FramePredictionRecord, TextRegion


class TextDetector(Protocol):
    """Detector interface needed by the video redaction workflow."""

    def predict(
        self,
        image: Any,
        *,
        image_path: str | None = None,
        frame_index: int | None = None,
    ) -> list[TextRegion]:
        """Return text regions for one video frame."""


@dataclass(frozen=True)
class VideoRedactionConfig:
    """Configuration for offline video text redaction."""

    input_video: Path
    output_root: Path = Path("runs/redactions")
    run_id: str | None = None
    mask_color: MaskColor = "black"
    expand_pixels: float = 0.0
    expand_ratio: float = 0.0
    codec: str = "mp4v"


@dataclass(frozen=True)
class VideoRedactionPaths:
    """Filesystem paths produced by one video redaction run."""

    run_dir: Path
    predictions_dir: Path
    redacted_dir: Path
    prediction_jsonl: Path
    redacted_video: Path
    summary_json: Path
    summary_csv: Path


@dataclass(frozen=True)
class VideoRedactionResult:
    """Summary of a completed video redaction run."""

    paths: VideoRedactionPaths
    frame_count: int
    text_region_count: int
    fps: float
    width: int
    height: int


def _import_cv2() -> Any:
    """Import OpenCV only when video workflows are executed."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install project dependencies before "
            "running video redaction."
        ) from exc
    return cv2


def _build_run_id(input_video: Path) -> str:
    """Build a stable-looking run id from timestamp and input stem."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{input_video.stem}"


def _build_output_paths(config: VideoRedactionConfig) -> VideoRedactionPaths:
    """Build all artifact paths for one video redaction run."""
    run_id = config.run_id or _build_run_id(config.input_video)
    run_dir = config.output_root / run_id
    predictions_dir = run_dir / "predictions"
    redacted_dir = run_dir / "redacted"
    redacted_video = redacted_dir / f"{config.input_video.stem}.mp4"
    return VideoRedactionPaths(
        run_dir=run_dir,
        predictions_dir=predictions_dir,
        redacted_dir=redacted_dir,
        prediction_jsonl=predictions_dir / "text_regions.jsonl",
        redacted_video=redacted_video,
        summary_json=run_dir / "summary.json",
        summary_csv=run_dir / "summary.csv",
    )


def _ensure_output_dirs(paths: VideoRedactionPaths) -> None:
    """Create output directories needed for video artifacts."""
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    paths.redacted_dir.mkdir(parents=True, exist_ok=True)


def _read_capture_property(
    video_capture: Any,
    property_id: int,
    fallback: float,
) -> float:
    """Read an OpenCV capture property with a fallback for invalid values."""
    property_value = float(video_capture.get(property_id) or 0.0)
    if property_value <= 0:
        return fallback
    return property_value


def _bgr_frame_to_rgb_image(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame to a Pillow RGB image."""
    rgb_frame = frame[:, :, ::-1]
    return Image.fromarray(rgb_frame)


def _rgb_image_to_bgr_frame(image: Image.Image) -> np.ndarray:
    """Convert a Pillow RGB image to an OpenCV BGR frame."""
    rgb_frame = np.asarray(image.convert("RGB"))
    return rgb_frame[:, :, ::-1]


def _open_video_capture(cv2_module: Any, input_video: Path) -> Any:
    """Open a video capture and fail clearly when the file cannot be read."""
    video_capture = cv2_module.VideoCapture(str(input_video))
    if not video_capture.isOpened():
        raise ValueError(f"Could not open input video: {input_video}")
    return video_capture


def _create_video_writer(
    cv2_module: Any,
    output_video: Path,
    *,
    codec: str,
    fps: float,
    width: int,
    height: int,
) -> Any:
    """Create an OpenCV video writer for redacted frames."""
    fourcc = cv2_module.VideoWriter_fourcc(*codec)
    video_writer = cv2_module.VideoWriter(
        str(output_video), fourcc, fps, (width, height)
    )
    if not video_writer.isOpened():
        raise ValueError(f"Could not open output video writer: {output_video}")
    return video_writer


def _write_frame_prediction(
    prediction_file: Any,
    *,
    input_video: Path,
    frame_index: int,
    width: int,
    height: int,
    regions: list[TextRegion],
) -> None:
    """Append one JSONL prediction record for a processed video frame."""
    prediction_record = FramePredictionRecord(
        image_path=str(input_video),
        frame_index=frame_index,
        width=width,
        height=height,
        regions=regions,
        metadata={"source": "video_frame"},
    )
    prediction_file.write(json.dumps(prediction_record.to_dict()) + "\n")


def _write_summary_files(
    paths: VideoRedactionPaths,
    *,
    input_video: Path,
    frame_count: int,
    text_region_count: int,
    fps: float,
    width: int,
    height: int,
    audio_preserved: bool,
) -> None:
    """Write JSON and CSV summaries for a completed video redaction run."""
    summary = {
        "input_video": str(input_video),
        "redacted_video": str(paths.redacted_video),
        "prediction_jsonl": str(paths.prediction_jsonl),
        "frame_count": frame_count,
        "text_region_count": text_region_count,
        "fps": fps,
        "width": width,
        "height": height,
        "audio_preserved": audio_preserved,
    }
    paths.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    with paths.summary_csv.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=list(summary.keys()))
        csv_writer.writeheader()
        csv_writer.writerow(summary)


def redact_video_offline(
    config: VideoRedactionConfig,
    detector: TextDetector,
    *,
    cv2_module: Any | None = None,
) -> VideoRedactionResult:
    """Process every video frame, redact detected text, and write artifacts."""
    cv2_runtime = cv2_module or _import_cv2()
    paths = _build_output_paths(config)
    _ensure_output_dirs(paths)

    video_capture = _open_video_capture(cv2_runtime, config.input_video)
    fps = _read_capture_property(video_capture, cv2_runtime.CAP_PROP_FPS, 30.0)
    width = int(
        _read_capture_property(video_capture, cv2_runtime.CAP_PROP_FRAME_WIDTH, 0.0)
    )
    height = int(
        _read_capture_property(video_capture, cv2_runtime.CAP_PROP_FRAME_HEIGHT, 0.0)
    )
    video_writer = _create_video_writer(
        cv2_runtime,
        paths.redacted_video,
        codec=config.codec,
        fps=fps,
        width=width,
        height=height,
    )

    frame_count = 0
    text_region_count = 0
    try:
        with paths.prediction_jsonl.open("w") as prediction_file:
            while True:
                frame_available, frame = video_capture.read()
                if not frame_available:
                    break

                text_regions = detector.predict(
                    frame,
                    image_path=str(config.input_video),
                    frame_index=frame_count,
                )
                redacted_image = apply_solid_redaction(
                    _bgr_frame_to_rgb_image(frame),
                    text_regions,
                    mask_color=config.mask_color,
                    expand_pixels=config.expand_pixels,
                    expand_ratio=config.expand_ratio,
                )
                video_writer.write(_rgb_image_to_bgr_frame(redacted_image))
                _write_frame_prediction(
                    prediction_file,
                    input_video=config.input_video,
                    frame_index=frame_count,
                    width=width,
                    height=height,
                    regions=text_regions,
                )
                frame_count += 1
                text_region_count += len(text_regions)
    finally:
        video_capture.release()
        video_writer.release()

    if frame_count == 0:
        raise ValueError(
            f"Input video contains no readable frames: {config.input_video}"
        )

    _write_summary_files(
        paths,
        input_video=config.input_video,
        frame_count=frame_count,
        text_region_count=text_region_count,
        fps=fps,
        width=width,
        height=height,
        audio_preserved=False,
    )
    return VideoRedactionResult(
        paths=paths,
        frame_count=frame_count,
        text_region_count=text_region_count,
        fps=fps,
        width=width,
        height=height,
    )
