"""Command line entrypoint for text redaction benchmark workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


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
        "--model-name",
        default="PP-OCRv5_server_det",
        help="PaddleOCR text detection model name.",
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
    return 0
