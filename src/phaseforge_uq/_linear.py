"""Independent Gaussian-coefficient regression with scalar noise quadrature.

y | beta,sigma ~ N(X beta, sigma^2 I); beta ~ N(0,diag(tau^2));
sigma ~ HalfNormal(noise_scale). Units must be consistent with the supplied design matrix.
The coefficients are marginalized analytically. Quadrature is in log(sigma),
including its Jacobian. Joint draws use the conditional Gaussian distribution.
"""

import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.optimize import brentq


def mm(a, b):
    """Small vector/matrix contraction without platform BLAS flag artifacts."""
    a, b = np.asarray(a), np.asarray(b)
    return (
        np.einsum("...i,i->...", a, b, optimize=False)
        if b.ndim == 1
        else np.einsum("...i,ij->...j", a, b, optimize=False)
    )


class Posterior:
    def __init__(self, X, y, tau, noise_scale=0.02, grid_size=1601, sigma_bounds=(1e-6, 1.0)):
        self.X, self.y, self.tau = np.asarray(X, float), np.asarray(y, float), np.asarray(tau, float)
        n, p = self.X.shape
        if self.y.shape != (n,) or self.tau.shape != (p,) or n == 0 or p == 0:
            raise ValueError("Invalid data dimensions")
        if (
            not all(np.isfinite(a).all() for a in (self.X, self.y, self.tau))
            or np.any(self.tau <= 0)
            or not np.isfinite(noise_scale)
            or noise_scale <= 0
        ):
            raise ValueError("Nonfinite data or nonpositive prior scale")
        if grid_size < 3 or not np.isfinite(sigma_bounds).all() or not 0 < sigma_bounds[0] < sigma_bounds[1]:
            raise ValueError("Invalid quadrature grid")
        self.noise_scale = noise_scale
        u, s, vt = np.linalg.svd(self.X * self.tau, full_matrices=True)
        k = len(s)
        self.s = np.pad(s, (0, p - k))
        self.vt = vt
        self.uy = np.pad(mm(u.T, self.y)[:k], (0, p - k))
        # An explicit orthogonal residual avoids cancellation near an exact fit.
        residual = mm(u.T, self.y)[k:]
        self.residual2 = float(mm(residual, residual))
        self.log_sigma = np.linspace(np.log(sigma_bounds[0]), np.log(sigma_bounds[1]), grid_size)
        self.sigma = np.exp(self.log_sigma)
        a = s[None, :] ** 2 + self.sigma[:, None] ** 2
        loglike = -0.5 * (
            np.log(a).sum(1)
            + (n - k) * 2 * self.log_sigma
            + ((self.uy[:k] ** 2) / a).sum(1)
            + self.residual2 / self.sigma**2
        )
        log_density = loglike - 0.5 * (self.sigma / noise_scale) ** 2 + self.log_sigma
        log_weight = log_density.copy()
        log_weight[[0, -1]] += np.log(0.5)
        self.weights = np.exp(log_weight - logsumexp(log_weight))
        self.boundary_mass = float(self.weights[:5].sum() + self.weights[-5:].sum())
        if self.boundary_mass > 1e-6:
            raise ValueError("Noise posterior reaches grid boundary; expand bounds")
        factors = self.s[None, :] / (self.s[None, :] ** 2 + self.sigma[:, None] ** 2)
        self.conditional_means = mm(factors * self.uy, vt) * self.tau
        self.conditional_variance_factors = self.sigma[:, None] ** 2 / (self.s[None, :] ** 2 + self.sigma[:, None] ** 2)
        self.mean = mm(self.weights, self.conditional_means)
        delta = self.conditional_means - self.mean
        transform = self.tau[:, None] * vt.T
        self.covariance = mm(transform * mm(self.weights, self.conditional_variance_factors), transform.T) + mm(
            delta.T * self.weights, delta
        )

    def conditional(self, sigma):
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive")
        mean = self.tau * mm(self.vt.T, self.s * self.uy / (self.s**2 + sigma**2))
        transform = self.tau[:, None] * self.vt.T
        cov = mm(transform * (sigma**2 / (self.s**2 + sigma**2)), transform.T)
        return mean, cov

    def sample(self, count=20000, seed=20261007):
        rng = np.random.default_rng(seed)
        index = rng.choice(len(self.sigma), count, p=self.weights)
        delta = rng.normal(size=(count, len(self.tau))) * np.sqrt(self.conditional_variance_factors[index])
        draws = self.conditional_means[index] + mm(delta, self.vt) * self.tau
        return draws, self.sigma[index]

    def intervals(self, rows, predictive=False, level=0.95):
        """Deterministic quantiles of the Gaussian mixture, no Monte Carlo error."""
        rows = np.atleast_2d(rows)
        means = mm(self.conditional_means, rows.T)
        projected = mm(rows * self.tau, self.vt.T)
        variances = mm(self.conditional_variance_factors, (projected**2).T)
        if predictive:
            variances += self.sigma[:, None] ** 2
        result = []
        for mu, var in zip(means.T, variances.T):
            if np.max(var) == 0:
                result.append([float(mu[0]), float(mu[0])])
                continue
            sd = np.sqrt(np.maximum(var, np.finfo(float).tiny))
            lo, hi = float(np.min(mu - 12 * sd)), float(np.max(mu + 12 * sd))

            def cdf(z):
                return float(mm(self.weights, ndtr((z - mu) / sd)))

            result.append(
                [brentq(lambda z: cdf(z) - q, lo, hi, xtol=1e-12) for q in ((1 - level) / 2, (1 + level) / 2)]
            )
        return np.asarray(result)
