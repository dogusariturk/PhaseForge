"""Public fixed-design fitting API; no equilibrium minimization is performed."""

from copy import deepcopy

import numpy as np

from phaseforge_uq._linear import Posterior, mm
from phaseforge_uq.bundle import write_bundle


class BayesianTdbFit:
    """Fit ``energy = offset + design @ coefficients + Gaussian noise``.

    Each column has an independent Normal prior with the supplied center and
    scale. A scalar HalfNormal residual scale is marginalized numerically;
    posterior coefficients are generally correlated. The caller constructs the
    fixed-site design and supplies all energy/unit conversions explicitly.
    Internal-equilibrium and multiphase observations are unsupported.
    """

    def __init__(
        self,
        design,
        energies,
        *,
        observation_kind,
        coefficient_ids,
        coefficient_units,
        energy_units,
        prior_mean,
        prior_scale,
        noise_scale,
        sigma_bounds,
        offset=None,
        grid_size=801,
    ):
        """Validate the observation contract and fit on two noise grids.

        Units must use J/mol_atoms for observations and either
        J/mol_formula_unit or J/mol_formula_unit/K for coefficients. Design
        columns must include site-ratio normalization and temperature factors.
        ``offset`` contains known terms only; the prior mean is added separately.
        A failed grid convergence check raises rather than returning a fit.
        """
        if observation_kind != "fixed_configuration":
            raise NotImplementedError("Only fixed_configuration observations are supported")
        if energy_units != "J/mol_atoms":
            raise ValueError("Convert observation energies to J/mol_atoms explicitly")
        matrix = np.array(design, dtype=float, copy=True)
        values = np.array(energies, dtype=float, copy=True)
        if matrix.ndim != 2 or min(matrix.shape) == 0 or values.shape != (matrix.shape[0],):
            raise ValueError("Expected nonempty n by p design and n energies")
        size = matrix.shape[1]
        ids = tuple(coefficient_ids)
        units = tuple(coefficient_units)
        if len(ids) != size or any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != size:
            raise ValueError("Unique coefficient IDs must match design columns")
        if len(units) != size or any(x not in ("J/mol_formula_unit", "J/mol_formula_unit/K") for x in units):
            raise ValueError("Declare each coefficient's formula-unit energy or entropy units")
        center = np.array(prior_mean, dtype=float, copy=True)
        scale = np.array(prior_scale, dtype=float, copy=True)
        known = np.zeros(len(values)) if offset is None else np.array(offset, dtype=float, copy=True)
        if center.shape != (size,) or scale.shape != (size,) or known.shape != values.shape:
            raise ValueError("Prior or offset dimensions do not match design")
        if not all(np.isfinite(x).all() for x in (matrix, values, center, scale, known)):
            raise ValueError("Inputs must be finite")
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 33:
            raise ValueError("grid_size must be an integer of at least 33")
        residual = values - known - mm(matrix, center)
        options = {"noise_scale": noise_scale, "sigma_bounds": sigma_bounds}
        coarse = Posterior(matrix, residual, scale, grid_size=grid_size, **options)
        fine = Posterior(matrix, residual, scale, grid_size=2 * grid_size - 1, **options)
        sd = np.sqrt(np.diag(fine.covariance))
        normalizer = sd[:, None] * sd[None, :]
        error = max(
            float(np.max(np.abs(fine.mean - coarse.mean) / sd)),
            float(np.max(np.abs(fine.covariance - coarse.covariance) / normalizer)),
        )
        if not np.isfinite(error) or error >= 1e-4:
            raise ValueError("Noise quadrature did not converge; increase grid_size and check sigma_bounds")
        self._posterior = fine
        self._center = center
        self.coefficient_ids = ids
        self.coefficient_units = units
        self.energy_units = energy_units
        self.diagnostics = {
            "grid_size": 2 * grid_size - 1,
            "normalized_moment_error": error,
            "boundary_mass": fine.boundary_mass,
        }
        self._inference = {
            "coefficient_prior": "independent Normal",
            "prior_mean": center.tolist(),
            "prior_scale": scale.tolist(),
            "residual_prior": "HalfNormal",
            "noise_scale": float(noise_scale),
            "sigma_bounds": list(sigma_bounds),
            "likelihood": "independent Gaussian with shared scalar residual scale",
        }

    @property
    def mean(self):
        """Return the posterior mean in declared coefficient order and units."""
        return self._center + self._posterior.mean

    @property
    def covariance(self):
        """Return the full coefficient covariance, retaining cross-correlations."""
        return self._posterior.covariance.copy()

    def sample(self, count=20000, seed=20261007):
        """Return joint coefficient draws and their paired residual scales."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        draws, sigmas = self._posterior.sample(count, seed)
        return draws + self._center, sigmas

    def intervals(self, design, *, offset=None, predictive=False, level=0.95):
        """Return latent or future-observation equal-tail energy intervals."""
        rows = np.atleast_2d(np.asarray(design, dtype=float))
        known = np.zeros(len(rows)) if offset is None else np.asarray(offset, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != len(self.coefficient_ids) or known.shape != (len(rows),):
            raise ValueError("Query design or offset dimensions are inconsistent")
        if not np.isfinite(rows).all() or not np.isfinite(known).all() or not 0 < level < 1:
            raise ValueError("Finite query inputs and 0 < level < 1 required")
        shift = known + mm(rows, self._center)
        return self._posterior.intervals(rows, predictive=predictive, level=level) + shift[:, None]

    def export(self, mean_tdb_text, metadata, output_dir, *, count=20000, seed=20261007, tdb_name="mean.tdb"):
        """Export a supplied mean TDB and full joint posterior sidecars.

        The caller renders the TDB using ``self.mean``. Its mapped coefficients
        and units are checked against the fit before any files are written.
        Existing output files are never overwritten.
        """
        meta = deepcopy(metadata)
        parameters = meta.get("parameters", [])
        if tuple(p.get("id") for p in parameters) != self.coefficient_ids:
            raise ValueError("TDB mapping order must match fitted coefficient IDs")
        if tuple(p.get("units") for p in parameters) != self.coefficient_units:
            raise ValueError("TDB coefficient units must match the fitted units")
        meta.update(
            observation_kind="fixed_configuration",
            residual_units=self.energy_units,
            inference=deepcopy(self._inference),
            quadrature=dict(self.diagnostics),
            sample_seed=seed,
            sample_count=count,
        )
        draws, sigmas = self.sample(count, seed)
        arrays = {
            "joint_draws": draws,
            "sigmas": sigmas,
            "mean": self.mean,
            "coefficient_ids": np.asarray(self.coefficient_ids),
        }
        return write_bundle(mean_tdb_text, meta, arrays, output_dir, tdb_name=tdb_name)
