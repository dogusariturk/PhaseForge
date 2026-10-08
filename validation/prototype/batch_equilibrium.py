"""Batched evaluation of the same certified CEF model used by the scalar solver."""
import itertools
import numpy as np
from .equilibrium_solver import EquilibriumModel, ConvergenceError, NonConvexError, R


class BoundedBatchEquilibrium:
    def __init__(self, composition, temperature, center, free_indices, bound):
        self.model = EquilibriumModel(composition, temperature)
        if not np.all(self.model.interior):
            raise ValueError("Batch benchmark requires interior compositions")
        self.center = self.model._theta(center).copy()
        self.free_indices = np.asarray(free_indices, int)
        if self.free_indices.shape != (2,) or len(set(self.free_indices)) != 2 or np.any((self.free_indices < 0) | (self.free_indices >= 10)):
            raise ValueError("Two distinct coefficient indices required")
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("Positive finite prior half-width required")
        self.bound = float(bound)
        self.corner_bounds = []
        for offsets in itertools.product([-self.bound, self.bound], repeat=2):
            coefficients = self.center.copy()
            coefficients[self.free_indices] += offsets
            self.corner_bounds.append(float(self.model.curvature_bound(coefficients).min()))
        if min(self.corner_bounds) <= 0:
            raise NonConvexError("Complete prior rectangle is not certified convex")
        self.base_polynomial = np.einsum("spk,p->sk", self.model.basis, self.center, optimize=False)

    def evaluate(self, shifts, chunk_size=512):
        shifts = np.asarray(shifts, float)
        if shifts.ndim != 2 or shifts.shape[1] != 2 or not np.isfinite(shifts).all() or np.any(abs(shifts) > self.bound):
            raise ValueError("Coefficient shifts must remain inside the certified rectangle")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("Positive integer chunk size required")
        outputs = []
        for first in range(0, len(shifts), chunk_size):
            offsets = shifts[first:first+chunk_size]
            polynomial = self.base_polynomial[None, :, :]+np.einsum(
                "bp,spk->bsk", offsets, self.model.basis[:, self.free_indices, :], optimize=False)
            bulk = self.model.x[None, :]
            temperature = self.model.T[None, :]
            sites = np.broadcast_to(bulk, polynomial.shape[:2]).copy()
            lower = np.broadcast_to(self.model.lo, sites.shape).copy()
            upper = np.broadcast_to(self.model.hi, sites.shape).copy()
            coefficients = np.broadcast_to(self.center, (len(offsets), 10)).copy()
            coefficients[:, self.free_indices] += offsets
            tolerance = 1e-8+1e-11*(R*temperature+np.max(abs(coefficients[:, :9]), axis=1)[:, None]+abs(coefficients[:, 9, None])*temperature)
            for iteration in range(100):
                other_sites = (4*bulk-sites)/3
                gradient = ((4*polynomial[:, :, 4]*sites+3*polynomial[:, :, 3])*sites+2*polynomial[:, :, 2])*sites+polynomial[:, :, 1]
                gradient += R*temperature/4*(np.log(sites)-np.log1p(-sites)-np.log(other_sites)+np.log1p(-other_sites))
                curvature = (12*polynomial[:, :, 4]*sites+6*polynomial[:, :, 3])*sites+2*polynomial[:, :, 2]
                curvature += R*temperature/4*(1/(sites*(1-sites))+1/(3*other_sites*(1-other_sites)))
                finished = abs(gradient) <= tolerance
                if np.all(finished):
                    break
                lower = np.where((gradient < 0) & ~finished, sites, lower)
                upper = np.where((gradient >= 0) & ~finished, sites, upper)
                proposal = sites-gradient/curvature
                valid = (proposal > lower) & (proposal < upper) & np.isfinite(proposal)
                sites = np.where(finished, sites, np.where(valid, proposal, (lower+upper)/2))
            else:
                raise ConvergenceError("Batched internal minimization did not converge")
            mixing = lambda fractions: fractions*np.log(fractions)+(1-fractions)*np.log1p(-fractions)
            energy = ((polynomial[:, :, 4]*sites+polynomial[:, :, 3])*sites+polynomial[:, :, 2])*sites**2+polynomial[:, :, 1]*sites+polynomial[:, :, 0]
            energy += R*temperature/4*(mixing(sites)+3*mixing(other_sites))
            if not np.isfinite(energy).all():
                raise ConvergenceError("Nonfinite batched energy")
            outputs.append(energy)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(self.model.x)))
