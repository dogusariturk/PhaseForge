"""CI-01: an empty suite cannot be reported as successful to CI consumers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestRunnerContract(unittest.TestCase):
    def test_empty_suite_fails_exit_status_and_json_report(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/"empty.json"
            process = subprocess.run(
                [sys.executable, str(root/"run_tests.py"), "--pattern", "no_tests_match_*.py", "--report", str(report)],
                cwd=directory, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(process.returncode, 1, process.stdout+process.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data["tests_run"], 0)
            self.assertIs(data["passed"], False)


if __name__ == "__main__":
    unittest.main()
