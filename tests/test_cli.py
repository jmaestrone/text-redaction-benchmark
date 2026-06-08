"""Smoke tests for the public CLI entrypoint."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from text_redaction_benchmark.cli import build_parser, main


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

    def test_main_accepts_empty_arguments(self) -> None:
        """Check that invoking the top-level command without args succeeds."""
        self.assertEqual(main([]), 0)

    def test_detect_text_execution_is_not_wired_yet(self) -> None:
        """Check that detect-text execution fails loudly until workflows exist."""
        with io.StringIO() as stderr:
            with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
                main(["detect-text", "sample.png"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
