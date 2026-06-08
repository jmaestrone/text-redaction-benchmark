"""Command line entrypoint for text redaction benchmark workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="text-redact",
        description=(
            "Offline text detection and redaction benchmark workflows. "
            "Model-backed commands will be added in later checkpoints."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="text-redact 0.1.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the text-redact CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0
