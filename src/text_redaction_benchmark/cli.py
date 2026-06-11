"""Command line entrypoint for text redaction benchmark workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from text_redaction_benchmark.paddleocr_detector import (
    PaddleOCRTextDetector,
    PaddleOCRTextDetectorConfig,
)
from text_redaction_benchmark.video import VideoRedactionConfig, redact_video_offline


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="text-redact",
        description=("Offline text detection and redaction benchmark workflows."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="text-redact 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command")
    detect_text_parser = subparsers.add_parser(
        "detect-text",
        help="Detect text regions in images with PaddleOCR.",
        description=(
            "Detect text regions with PaddleOCR text detection. "
            "Image execution will be wired into artifact-writing workflows in later checkpoints."
        ),
    )
    detect_text_parser.add_argument(
        "image",
        nargs="?",
        help="Image path to process in a later checkpoint.",
    )
    detect_text_parser.add_argument(
        "--detector",
        choices=("paddleocr", "rfdetr"),
        default="paddleocr",
        help="Text detector backend. PaddleOCR remains the default.",
    )
    detect_text_parser.add_argument(
        "--model-name",
        default="PP-OCRv5_server_det",
        help="PaddleOCR text detection model name.",
    )
    detect_text_parser.add_argument(
        "--rfdetr-model-size",
        choices=("nano", "small", "medium", "large"),
        default="medium",
        help="RF-DETR model size for the box-based text baseline.",
    )
    detect_text_parser.add_argument(
        "--rfdetr-threshold",
        type=float,
        default=0.5,
        help="RF-DETR detection threshold.",
    )
    detect_text_parser.add_argument(
        "--rfdetr-text-class-id",
        type=int,
        default=0,
        help="RF-DETR class id treated as text. Use one-class text models by default.",
    )
    detect_text_parser.add_argument(
        "--device",
        default=None,
        help='PaddleOCR device string, such as "cpu" or "gpu:0".',
    )
    detect_text_parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Pixel score threshold for text pixels.",
    )
    detect_text_parser.add_argument(
        "--box-thresh",
        type=float,
        default=None,
        help="Detected box score threshold.",
    )
    detect_text_parser.add_argument(
        "--unclip-ratio",
        type=float,
        default=None,
        help="PaddleOCR text region expansion ratio.",
    )
    detect_text_parser.add_argument(
        "--limit-side-len",
        type=int,
        default=None,
        help="Input image side length limit for detection.",
    )
    redact_image_parser = subparsers.add_parser(
        "redact-image",
        help="Apply solid masks to detected text regions in an image.",
        description=(
            "Apply solid polygon masks to detected text regions in a local image. "
            "Prediction input and artifact writing will be wired into later workflows."
        ),
    )
    redact_image_parser.add_argument(
        "image",
        nargs="?",
        help="Image path to redact in a later checkpoint.",
    )
    redact_image_parser.add_argument(
        "--mode",
        choices=("solid",),
        default="solid",
        help="Redaction mode. Solid masking is the privacy-first default.",
    )
    redact_image_parser.add_argument(
        "--mask-color",
        choices=("black", "white"),
        default="black",
        help="Solid mask color.",
    )
    redact_image_parser.add_argument(
        "--expand-pixels",
        type=float,
        default=0.0,
        help="Absolute polygon expansion in pixels before masking.",
    )
    redact_image_parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.0,
        help="Relative polygon expansion ratio before masking.",
    )
    redact_video_parser = subparsers.add_parser(
        "redact-video",
        help="Redact detected text in every frame of a local video.",
        description=(
            "Process every frame offline with PaddleOCR text detection, apply solid "
            "text masks, and write redaction artifacts under runs/redactions."
        ),
    )
    redact_video_parser.add_argument(
        "video",
        help="Input video path.",
    )
    redact_video_parser.add_argument(
        "--output-root",
        default="runs/redactions",
        help="Root directory for redaction run artifacts.",
    )
    redact_video_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Defaults to a timestamp plus input video stem.",
    )
    redact_video_parser.add_argument(
        "--mask-color",
        choices=("black", "white"),
        default="black",
        help="Solid mask color.",
    )
    redact_video_parser.add_argument(
        "--expand-pixels",
        type=float,
        default=0.0,
        help="Absolute polygon expansion in pixels before masking.",
    )
    redact_video_parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.0,
        help="Relative polygon expansion ratio before masking.",
    )
    redact_video_parser.add_argument(
        "--model-name",
        default="PP-OCRv5_server_det",
        help="PaddleOCR text detection model name.",
    )
    redact_video_parser.add_argument(
        "--device",
        default=None,
        help='PaddleOCR device string, such as "cpu" or "gpu:0".',
    )
    redact_video_parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Pixel score threshold for text pixels.",
    )
    redact_video_parser.add_argument(
        "--box-thresh",
        type=float,
        default=None,
        help="Detected box score threshold.",
    )
    redact_video_parser.add_argument(
        "--unclip-ratio",
        type=float,
        default=None,
        help="PaddleOCR text region expansion ratio.",
    )
    redact_video_parser.add_argument(
        "--limit-side-len",
        type=int,
        default=None,
        help="Input image side length limit for detection.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the text-redact CLI."""
    parser = build_parser()
    parsed_args = parser.parse_args(argv)
    if parsed_args.command == "detect-text":
        parser.error(
            "detect-text execution will be wired into artifact-writing workflows "
            "in a later checkpoint; use --help to inspect options."
        )
    if parsed_args.command == "redact-image":
        parser.error(
            "redact-image execution will be wired into artifact-writing workflows "
            "in a later checkpoint; use --help to inspect options."
        )
    if parsed_args.command == "redact-video":
        detector_config = PaddleOCRTextDetectorConfig(
            model_name=parsed_args.model_name,
            device=parsed_args.device,
            thresh=parsed_args.thresh,
            box_thresh=parsed_args.box_thresh,
            unclip_ratio=parsed_args.unclip_ratio,
            limit_side_len=parsed_args.limit_side_len,
        )
        video_config = VideoRedactionConfig(
            input_video=Path(parsed_args.video),
            output_root=Path(parsed_args.output_root),
            run_id=parsed_args.run_id,
            mask_color=parsed_args.mask_color,
            expand_pixels=parsed_args.expand_pixels,
            expand_ratio=parsed_args.expand_ratio,
        )
        result = redact_video_offline(
            video_config, PaddleOCRTextDetector(detector_config)
        )
        print(result.paths.run_dir)
    return 0
