"""Resolve the archived calibration datasets with full-support focused quadrature."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaigns.nonlinear_calibration import CENTER, FREE, SD, BOUND, S0, calibration_tests, converged, exit_status, save
from prototype.adaptive_quadrature import integrate
from prototype.batch_equilibrium import BoundedBatchEquilibrium
from prototype.noise_marginal import log_marginal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    arguments = parser.parse_args()
    baseline = arguments.baseline.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    specification = json.loads((baseline/"specification.json").read_text())
    with np.load(baseline/"simulated_datasets.npz", allow_pickle=False) as archive:
        truths = archive["coefficient_shifts"].copy()
        observations = archive["observations"].copy()
        sigmas = archive["sigmas"].copy()
    indices = list(range(len(truths))) if arguments.indices is None else arguments.indices
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(truths) for index in indices):
        raise ValueError("Dataset indices must be distinct and valid")
    model = BoundedBatchEquilibrium(specification["observation_x"], specification["observation_T"], CENTER, FREE, BOUND)
    source_files = [Path(__file__), ROOT/"campaigns/nonlinear_calibration.py", *sorted((ROOT/"prototype").glob("*.py"))]
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        baseline_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(baseline.glob("*")) if path.is_file()},
        environment=dict(python=platform.python_version(), machine=platform.machine(), dependencies={name: importlib.metadata.version(name) for name in ["numpy", "scipy"]}),
        indices=indices, entire_campaign=indices == list(range(len(truths))),
        method="Tensor Simpson quadrature in sinh coordinates with explicit Jacobian, centered/scaled by archived posterior moments; full prior support; exact piecewise-linear marginal integration and inversion",
        node_levels=[33, 65, 129, 257, 513], reference="Doubled axis scales and shifted centers, twice selected grid resolution minus one",
        acceptance="Original CDF .005, mean .01 SD, relative SD .01, credible endpoint .03 SD limits plus log-integral change < .001; both refinement and displaced-reference comparisons required",
        corner_curvature_bounds=model.corner_bounds,
        scope="Same 64 manufactured datasets; two uncertain coefficients; eight fixed; convex internal equilibrium; coefficient calibration only",
    )
    save(output/"specification.json", manifest)
    records = []
    bounds = np.array([[-BOUND, BOUND], [-BOUND, BOUND]])
    for index in indices:
        record_start = time.monotonic()
        pilot = json.loads((baseline/f"replicate_{index:04d}.json").read_text())["selected"]
        center = np.asarray(pilot["mean"])
        scales = np.asarray(pilot["sd"])
        def log_density(points):
            predictions = model.evaluate(points)
            residual_sum = np.sum((predictions-observations[index])**2, axis=1)
            return log_marginal(residual_sum, observations.shape[1], S0)-.5*np.sum((points/SD)**2, axis=1)
        levels = []
        accepted = False
        reference = None
        for count in [33, 65, 129, 257, 513]:
            current = integrate(log_density, bounds, center, scales, truths[index], count)
            levels.append(current)
            if len(levels) < 2:
                continue
            comparison = converged(levels[-2], current)
            comparison["log_integral_difference"] = abs(levels[-2]["log_integral"]-current["log_integral"])
            if not comparison["passes"] or comparison["log_integral_difference"] >= .001:
                continue
            reference = integrate(log_density, bounds, center+scales*np.array([.31, -.27]), scales*2, truths[index], 2*count-1)
            reference_comparison = converged(current, reference)
            reference_comparison["log_integral_difference"] = abs(current["log_integral"]-reference["log_integral"])
            cuts = [.025, .25, .5, .75, .975]
            reference_comparison["calibration_bins_stable"] = bool(np.array_equal(np.searchsorted(cuts, current["cdfs"]), np.searchsorted(cuts, reference["cdfs"])))
            accepted = reference_comparison["passes"] and reference_comparison["log_integral_difference"] < .001 and reference_comparison["calibration_bins_stable"]
            if accepted:
                break
        record = dict(index=index, sigma=float(sigmas[index]), pilot_center=center.tolist(), pilot_scale=scales.tolist(), levels=levels,
                      comparison=comparison, reference=reference,
                      reference_comparison=reference_comparison if reference is not None else None,
                      accepted=bool(accepted), selected=reference if accepted else current, seconds=time.monotonic()-record_start)
        records.append(record)
        save(output/f"replicate_{index:04d}.json", record)
        progress = dict(completed=len(records), total=len(indices), index=index, accepted=bool(accepted), seconds=time.monotonic()-started)
        save(output/"progress.json", progress)
        print(json.dumps(progress), flush=True)
    unresolved = [record["index"] for record in records if not record["accepted"]]
    entire = indices == list(range(len(truths)))
    calibration = calibration_tests([record["selected"]["cdfs"] for record in records]) if entire and not unresolved else None
    status = "inconclusive_grid_resolution" if unresolved else ("numerical_subset_resolved" if not entire else ("calibration_rejected" if calibration["rejected"] else "no_calibration_failure_detected"))
    summary = dict(status=status, all_numerically_resolved=not unresolved, entire_campaign=entire,
                   replicates=len(records), unresolved_replicates=unresolved, calibration=calibration,
                   seconds=time.monotonic()-started,
                   interpretation="Non-rejection for 64 prior-predictive datasets has limited power and does not establish physical calibration, predictive-energy coverage, or a general nonlinear solver.")
    save(output/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if status == "numerical_subset_resolved" else exit_status(status)


if __name__ == "__main__":
    raise SystemExit(main())
