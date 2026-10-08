# Bayesian fitting of fixed-configuration energies

This optional development feature fits an explicitly supplied linear Gibbs
model to fixed-configuration energies. Install it separately from the existing
CMake tools:

```sh
python3 -m pip install .
phaseforge-uq examples/uq/unequal_substitutional/fit.json --output uq-result
```

The output is `mean.tdb`, `mean.uq.json`, and `mean.uq.npz`. Keep all three files
together. The examples are manufactured software fixtures, not assessments of
Al–Ni or Fe–Ni–C. The second example, `interstitial_vacancy`, exercises
composition-dependent atom normalization with vacancies.

## Model and observations

The model is `energy = offset + design @ coefficients + residual`.
The JSON `fit` object declares the design matrix, observations, fixed offset,
coefficient IDs and units, prior centers and scales, residual prior scale, and
numerical noise integration bounds. `observation_kind` must be
`fixed_configuration`. SQS energies retain their prescribed occupations; the
code does not minimize site fractions or convert them into equilibrium data.

Observations and residual scales use J/mol atoms. Coefficients use J/mol
formula unit, or J/mol formula unit/K for a linear temperature coefficient.
The caller must include the correct temperature and occupied-site normalization
in the design matrix. In particular, vacancies do not contribute atoms to the
normalization. Unit labels cannot verify the numerical construction of an
arbitrary user-supplied matrix. Independent fixture tests check these conversions
against pycalphad for the two supplied examples.

`offset` contains known terms, such as an ideal-mixing term when fitting a Gibbs
energy. It excludes the contribution from the prior mean, which the fitter adds
itself. For static configuration energies, do not automatically add a thermal
ideal-mixing contribution to the observations.

Each coefficient has an independent Normal prior. Residuals are conditionally
independent Gaussian with one shared standard deviation, whose prior is
HalfNormal. The coefficient posterior is a correlated Gaussian mixture after
integrating over the residual scale. Noise integration uses log coordinates,
including the Jacobian; the bounds approximate the positive half-line.
Boundary mass and successive-grid moment checks must pass. A failed check
raises an error; increase grid resolution or extend bounds as appropriate.

These intervals describe uncertainty conditional on the supplied basis,
priors, observations and residual model. They do not quantify all MLIP error,
missing thermal physics or model-form uncertainty. Latent intervals exclude
future residual noise; `predictive=True` includes it.

## Python API and export

```python
from phaseforge_uq import BayesianTdbFit, load_bundle

fit = BayesianTdbFit(**fit_configuration)
mean = fit.mean
covariance = fit.covariance
draws, residual_scales = fit.sample(count=20000, seed=42)
# Render the selected Gibbs model with mean, preserving coefficient order.
tdb = fit.export(mean_tdb_text, metadata, "new-output", seed=42)
text, metadata, arrays = load_bundle(tdb)
```

`metadata.parameters` maps each column to its full TDB identity: phase,
parameter type, ordered constituent array, interaction order and polynomial
term. Units and coefficient order must match the fit. The CLI uses a TDB
template containing literal `{{coefficient_id}}` placeholders; it performs
numeric substitution without evaluating Python expressions. Template paths
are relative to the JSON input file.

Export checks that mapped TDB expressions reproduce the fitted mean.
Version 1 supports explicit constant and linear-temperature coefficients;
mapped named functions and higher temperature powers are rejected. Fixed,
unmapped terms may remain. The bundle stores complete joint draws and paired
residual scales, preserving correlation without reducing it to independent
error bars. JSON records priors, units, maps, quadrature diagnostics and seed.

TDB `$` comment directives point to the JSON, which points to the NPZ.
Checksums bind the three files; loading rejects missing files, unsafe relative
paths, mismatched arrays and inconsistent parameter maps. Native pycalphad
reads the mean TDB and ignores the comments. Its ordinary database re-export
may remove those comments; use the bundle loader to retain uncertainty.
Checksums detect accidental changes, not replacement of an entire bundle.
Existing bundle files are never overwritten.

## Scope

This command is an opt-in fitting/export layer. It does not replace ATAT's
structure selection, construct a model basis from `terms.in`, or change
`sqscal`, MLIP relaxation, CMake installation, or deterministic TDB export.
The Python installation supplies `phaseforge-uq`; CMake alone does not.

Internal-equilibrium and multiphase likelihoods are rejected by the public
fitter. The nonlinear solvers under `validation/prototype` are manufactured
reference experiments with explicitly restricted models. They are not part of
the installed API. Passing their tests does not establish a general nonlinear
CALPHAD calibration capability.

See [validation and reproducibility](../validation/README.md) for test commands
and the distinction between software checks and statistical calibration.
