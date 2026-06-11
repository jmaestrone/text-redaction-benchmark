"""Smoke tests for the public CLI entrypoint."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from text_redaction_benchmark.cli import build_parser, main
from text_redaction_benchmark.video import VideoRedactionPaths, VideoRedactionResult


class CliSmokeTest(unittest.TestCase):
    def test_help_includes_registered_command_name(self) -> None:
        """Check that top-level help includes the registered command name."""
        parser = build_parser()

        with io.StringIO() as stdout:
            with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
                parser.parse_args(["--help"])
            output = stdout.getvalue()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("text-redact", output)

    def test_detect_text_help_includes_paddleocr_options(self) -> None:
        """Check that detect-text help exposes PaddleOCR configuration knobs."""
        parser = build_parser()

        with io.StringIO() as stdout:
            with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
                parser.parse_args(["detect-text", "--help"])
            output = stdout.getvalue()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("detect-text", output)
        self.assertIn("--model-name", output)
        self.assertIn("--box-thresh", output)

    def test_redact_image_help_includes_solid_mask_options(self) -> None:
        """Check that redact-image help exposes solid masking options."""
        parser = build_parser()

        with io.StringIO() as stdout:
            with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
                parser.parse_args(["redact-image", "--help"])
            output = stdout.getvalue()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("redact-image", output)
        self.assertIn("--mask-color", output)
        self.assertIn("--expand-pixels", output)

    def test_redact_video_help_includes_offline_video_options(self) -> None:
        """Check that redact-video help exposes offline video options."""
        parser = build_parser()

        with io.StringIO() as stdout:
            with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
                parser.parse_args(["redact-video", "--help"])
            output = stdout.getvalue()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("redact-video", output)
        self.assertIn("--output-root", output)
        self.assertIn("--mask-color", output)
        self.assertIn("--model-name", output)

    def test_main_accepts_empty_arguments(self) -> None:
        """Check that invoking the top-level command without args succeeds."""
        self.assertEqual(main([]), 0)

    def test_detect_text_execution_is_not_wired_yet(self) -> None:
        """Check that detect-text execution fails loudly until workflows exist."""
        with io.StringIO() as stderr:
            with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
                main(["detect-text", "sample.png"])

        self.assertEqual(raised.exception.code, 2)

    def test_redact_video_execution_uses_detector_and_video_workflow(self) -> None:
        """Check that redact-video delegates execution to the video workflow."""
        result_paths = VideoRedactionPaths(
            run_dir=Path("runs/redactions/test-run"),
            predictions_dir=Path("runs/redactions/test-run/predictions"),
            redacted_dir=Path("runs/redactions/test-run/redacted"),
            prediction_jsonl=Path(
                "runs/redactions/test-run/predictions/text_regions.jsonl"
            ),
            redacted_video=Path("runs/redactions/test-run/redacted/input.mp4"),
            summary_json=Path("runs/redactions/test-run/summary.json"),
            summary_csv=Path("runs/redactions/test-run/summary.csv"),
        )
        result = VideoRedactionResult(
            paths=result_paths,
            frame_count=1,
            text_region_count=0,
            fps=30.0,
            width=2,
            height=2,
        )

        with (
            patch(
                "text_redaction_benchmark.cli.PaddleOCRTextDetector"
            ) as detector_class,
            patch(
                "text_redaction_benchmark.cli.redact_video_offline",
                return_value=result,
            ) as redact_video,
            io.StringIO() as stdout,
            redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "redact-video",
                    "input.mp4",
                    "--run-id",
                    "test-run",
                    "--mask-color",
                    "white",
                ]
            )
            output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        detector_class.assert_called_once()
        redact_video.assert_called_once()
        self.assertIn("runs/redactions/test-run", output)

    def test_redact_image_execution_is_not_wired_yet(self) -> None:
        """Check that redact-image execution fails loudly until workflows exist."""
        with io.StringIO() as stderr:
            with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
                main(["redact-image", "sample.png"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
