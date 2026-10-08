#!/usr/bin/env python3
"""Run portable tests from any directory and write a machine-readable report."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest
import warnings

ROOT = Path(__file__).resolve().parent
for directory in (ROOT/"prototype", ROOT/"fixtures", ROOT/"tests", ROOT/"campaigns"):
    sys.path.insert(0, str(directory))


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def startTest(self, test):
        self.started = time.perf_counter()
        self.current_outcome = "passed"
        super().startTest(test)

    def addError(self, test, err):
        self.current_outcome = "error"
        super().addError(test, err)

    def addFailure(self, test, err):
        self.current_outcome = "failed"
        super().addFailure(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.current_outcome = "failed"
        super().addSubTest(test, subtest, err)

    def addSkip(self, test, reason):
        self.current_outcome = "skipped: "+reason
        super().addSkip(test, reason)

    def stopTest(self, test):
        self.records.append(dict(id=test.id(), outcome=self.current_outcome,
                                 seconds=round(time.perf_counter()-self.started, 6)))
        super().stopTest(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py", help="unittest filename glob")
    parser.add_argument("--report", type=Path, default=ROOT/"reports"/"latest.json")
    args = parser.parse_args()
    warnings.filterwarnings("error", category=RuntimeWarning)
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"), pattern=args.pattern)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), platform=platform.platform(),
        dependencies={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "sympy", "pycalphad"]},
        command_arguments=sys.argv[1:],
        calibration_replicates_override=os.environ.get("PHASEFORGE_CALIBRATION_REPS"),
        tests_run=result.testsRun, passed=result.wasSuccessful() and result.testsRun > 0, skipped=len(result.skipped),
        seconds=round(time.perf_counter()-start, 3), tests=result.records,
        failures=[dict(test=t.id(), detail=trace) for t, trace in result.failures+result.errors],
        tested_target="Installed PhaseForge linear/backend bundle plus manufactured nonlinear references",
        public_cli_tested_by_this_suite=False,
        installed_linear_backend_tested="phaseforge_uq._linear" in sys.modules,
        general_nonconvex_equilibrium_fitting_supported=False,
        multiphase_likelihood_fitting_supported=False,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        imported_backend_files={name: str(Path(sys.modules[name].__file__).resolve())
                                for name in ["phaseforge_uq._linear", "phaseforge_uq.bundle", "bayes_linear", "cef_fixtures", "equilibrium_solver", "noise_marginal", "uq_bundle", "observation_contract"]
                                if name in sys.modules},
        installed_backend_sha256={name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
                                  for name in ["phaseforge_uq._linear", "phaseforge_uq.bundle"] if name in sys.modules},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ["prototype", "tests", "fixtures", "campaigns"]
                       for p in sorted((ROOT/folder).glob("*.py"))},
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(f"\nReport: {args.report}")
    return 0 if result.wasSuccessful() and result.testsRun > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
