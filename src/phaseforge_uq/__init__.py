"""Opt-in Bayesian fitting of fixed-configuration, linear Gibbs models."""

from phaseforge_uq.bundle import BundleError, load_bundle
from phaseforge_uq.fit import BayesianTdbFit

__all__ = ["BayesianTdbFit", "BundleError", "load_bundle"]
