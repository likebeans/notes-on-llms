import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"


class CliTests(unittest.TestCase):
    def run_cli(self, *args, seed="1"):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run([sys.executable, str(MAIN), *args], cwd=cwd,
                                    capture_output=True, text=True, timeout=10,
                                    env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(list(Path(cwd).iterdir()), [])
        return result

    def test_ask_works_outside_repository_and_outputs_valid_json(self):
        result = self.run_cli("ask", "借阅期限是多久？", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "answered")
        self.assertIn("21 天", data["answer"])

    def test_evaluation_is_reproducible_across_hash_seeds(self):
        first = self.run_cli("evaluate", "--json", seed="1")
        second = self.run_cli("evaluate", "--json", seed="2")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(first.stdout), json.loads(second.stdout))
        self.assertEqual(json.loads(first.stdout)["summary"]["count"], 20)

    def test_empty_question_reports_validation_error_with_nonzero_exit(self):
        result = self.run_cli("ask", "", "--json")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "error")

    def test_demo_includes_workflow_and_offline_report(self):
        result = self.run_cli("demo")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("read_evidence", result.stdout)
        self.assertIn("20", result.stdout)
        self.assertIn("invalid_arguments", result.stdout)
