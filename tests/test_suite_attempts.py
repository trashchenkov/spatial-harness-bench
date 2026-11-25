"""No model/API credentials needed to exercise evaluator-local attempt evidence."""

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_harness_bench import suite_adapter as adapter
from spatial_harness_bench import suite_native as native
from spatial_harness_bench.suite_attempt import LOG_LIMIT, Attempt


class AttemptTests(unittest.TestCase):
    def test_retries_have_separate_private_redacted_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"STAND_TEST_TOKEN": "demo-private-value"}):
            first = Attempt(tmp, family="building", mode="file", stage="file")
            second = Attempt(tmp, family="building", mode="file", stage="file")
            first.capture("stdout.log", b'Bearer invisible-bearer\n{"apiKey":"unknown-config-value"}\ndemo-private-value')
            second.capture("stderr.log", b"Authorization: Basic unknown-basic\nX-Api-Key: ignored-key")
            first.finish(75)
            second.finish(0)
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(stat.S_IMODE(first.path.stat().st_mode), 0o700)
            for path in (first.path / "stdout.log", second.path / "stderr.log", first.path / "attempt.json"):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                text = path.read_text()
                for secret in ("demo-private-value", "invisible-bearer", "unknown-config-value",
                               "unknown-basic", "ignored-key"):
                    self.assertNotIn(secret, text)
            self.assertEqual(json.loads((first.path / "attempt.json").read_text())["returncode"], 75)
            self.assertEqual(json.loads((second.path / "attempt.json").read_text())["returncode"], 0)

    def test_log_size_limit_and_partial_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"STAND_TEST_TOKEN": "demo-private-value"}):
            attempt = Attempt(tmp, family="building", mode="file", stage="file")
            attempt.capture("stdout.log", b"x" * (LOG_LIMIT * 2))
            attempt.capture("partial.txt", b"answer text demo-priv", truncated=True)
            self.assertEqual((attempt.path / "stdout.log").stat().st_size, LOG_LIMIT)
            self.assertTrue(attempt.record["streams"]["stdout.log"]["truncated"])
            self.assertNotIn("demo-priv", (attempt.path / "partial.txt").read_text())
            attempt.capture("headers.log", b'["Authorization","Basic test-key"]\n{"refresh_token":"partial-value',
                            truncated=True)
            header_text = (attempt.path / "headers.log").read_text()
            self.assertNotIn("test-key", header_text)
            self.assertNotIn("partial-value", header_text)

    def test_timeout_and_nonzero_exit_retain_native_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Attempt(tmp, family="building", mode="file", stage="file")
            with self.assertRaises(subprocess.TimeoutExpired):
                adapter.bounded_run([sys.executable, "-c",
                                     "import time; print('partial', flush=True); time.sleep(5)"], b"", 0.8, a)
            self.assertEqual((a.path / "stdout.log").read_text(), "partial\n")
            self.assertTrue(a.record["streams"]["stdout.log"]["truncated"])
            b = Attempt(tmp, family="building", mode="file", stage="file")
            proc = adapter.bounded_run([sys.executable, "-c",
                                        "import sys; print('failure output', file=sys.stderr); sys.exit(3)"],
                                       b"", 5, b)
            self.assertEqual(proc.returncode, 3)
            self.assertEqual((b.path / "stderr.log").read_text(), "failure output\n")

    def test_source_saves_answer_before_using_empty_artifact_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            task.mkdir()
            attempt = Attempt(root / "attempts", family="svg_recall", mode="source", stage="draw")

            def local_worker(*_args, **kwargs):
                return adapter.bounded_run([sys.executable, "-c", "print('not valid JSON')"], b"", 5,
                                           kwargs["attempt"])

            with patch.object(adapter, "invoke", side_effect=local_worker), patch.object(
                    adapter, "source_request", return_value=("draw", 2, "unchanged prompt")), patch.object(
                    adapter.Path, "cwd", return_value=task), patch("sys.stdin"):
                self.assertEqual(adapter.execute(attempt, "mock", "none", "10", "id",
                                                 "svg_recall", "source", "pi", "mock", "mock"), 0)
            self.assertEqual((attempt.path / "stdout.log").read_text(), "not valid JSON\n")
            self.assertEqual(attempt.record["parser"]["reason"], "invalid_json")
            self.assertTrue(attempt.record["parser"]["empty_artifact_fallback"])
            self.assertEqual((task / "desc.txt").read_bytes(), b"\n")
            self.assertEqual((task / "0.svg").read_bytes(), b"")

    def test_pi_text_survives_rpc_stream_aborting(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Attempt(tmp, family="svg_recall", mode="source", stage="draw")
            fixture = Path(tmp) / "fixture.py"
            fixture.write_text(
                "import json, sys\n"
                "sys.stdin.readline()\n"
                "event={'type':'message_update','assistantMessageEvent':"
                "{'type':'text_delta','delta':'partial SVG reply'}}\n"
                "print(json.dumps(event), flush=True)\n"
            )
            proc = adapter.bounded_run([sys.executable, native.__file__, "source", "pi", "", "5",
                                        sys.executable, str(fixture)], b"draw", 10, attempt)
            self.assertEqual(proc.returncode, 70)
            self.assertEqual((attempt.path / "partial-answer.txt").read_text(), "partial SVG reply")
            self.assertIn("rpc_closed", (attempt.path / "stderr.log").read_text())
            self.assertNotIn("partial SVG reply", adapter.ordinary_stderr(proc.stderr).decode())
