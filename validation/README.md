# Bayesian UQ validation

Install the package before testing. The production linear fitter and bundle
implementation are imported from `phaseforge_uq`, including by the compatibility
modules in `prototype/`. Other prototype modules are manufactured reference
solvers and fixture generators, excluded from the installed package.

```sh
python -m pip install . pytest
python -m pytest --junitxml=public-tests.xml
python validation/run_tests.py --report reference-tests.json
PHASEFORGE_CALIBRATION_REPS=1000 python validation/run_tests.py \
  --pattern test_calibration.py --report calibration-1000.json
```

The public tests run the actual CLI in a separate working directory for both
sublattice fixtures, then move the resulting bundle, load it, and verify joint
samples, noise scales and predictions. Negative tests cover unsupported
observation semantics, units, priors, IDs, sample counts, and export maps.

The reference suite covers analytic conditional posteriors, marginalized noise,
rank deficiency and identifiability, formula-unit/atom normalization,
pycalphad agreement, thermodynamic branches, coexistence oracles, bundle
integrity, fixed-site calibration, nonlinear campaign acceptance logic, and
adaptive quadrature. Runtime warnings are errors. An empty suite cannot pass.
The machine-readable report distinguishes the reference checks from public API
testing; use it together with the public test report.

`ci/uq-tests.yml` is a GitHub Actions template for Python 3.12 and 3.13, including
uploaded test reports. It is not active. The publishing credential lacks the
GitHub `workflow` scope; a maintainer can enable CI by copying this template to
`.github/workflows/uq-tests.yml` with an appropriately authorized credential.

The default fixed-site calibration is an 80-replicate screening test; the
1000-replicate command is the release check. Seeds and statistical acceptance
criteria are embedded in `tests/test_calibration.py`. It tests calibration
under the manufactured model, not accuracy of any MLIP or physical assessment.

## Nonlinear reference campaign

The supplied scripts retain the earlier two-coefficient internal-equilibrium
experiment: `(Al,Ni)1:(Al,Ni)3`, two uncertain coefficients, eight fixed,
bounded Gaussian coefficient priors, and a strict convexity certificate over
the full prior rectangle. They do not fit general nonconvex or multiphase data.

Generate the 64-dataset uniform-grid baseline and then refine it:

```sh
python validation/campaigns/nonlinear_calibration.py \
  --replicates 64 --seed 202610073 --max-grid 641 --output baseline64
python validation/campaigns/adaptive_nonlinear_calibration.py \
  --baseline baseline64 --output adaptive64
```

The uniform-grid run may exit nonzero for unresolved numerical integration;
its preserved dataset and pilot results are the adaptive run's inputs.

The adaptive script consumes all original simulated datasets and pilot
moments. It uses full-support sinh-coordinate Simpson integration with its
Jacobian and exact integration/inversion of piecewise-linear marginal PDFs.
Convergence requires successive-grid agreement in CDFs, means, standard
deviations and interval endpoints, plus a displaced, widened reference grid.
Unresolved datasets make the whole calibration inconclusive. The prior,
likelihood and statistical thresholds are unchanged by refinement.

Before integration, all 64 datasets passed numerical convergence and the
predeclared calibration tests. This historical result is not a new validation
of the installed fixed-site API. The campaign can be rerun from the shipped
source; full nonlinear campaigns are not required by ordinary CI.

`prototype/PROVENANCE.json` and `tests/LEGACY_PROVENANCE.json` record original
standalone sources. Linear inference and bundle handling have since moved
into the installed namespace; formatting and public-contract checks were
added. The hashes describe those original files, not the current wrappers.

The repository initially supplied no deterministic test suite. This addition
does not modify existing `bin/`, C++ sources, CMake configuration or submodule
references. An actual MLIP/ATAT workflow regression remains a separate release
check if subsequent integration changes that path.
