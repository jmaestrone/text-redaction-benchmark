"""Smoke tests for the public CLI entrypoint."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from text_redaction_benchmark.cli import build_parser, main


class CliSmokeTest(unittest.TestCase):
    def test_help_includes_registered_command_name(self) -> None:
        parser = build_parser()

        with io.StringIO() as stdout:
            with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
                parser.parse_args(["--help"])
            output = stdout.getvalue()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("text-redact", output)

    def test_main_accepts_empty_arguments(self) -> None:
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
