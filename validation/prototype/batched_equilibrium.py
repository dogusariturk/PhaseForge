"""Chunked convex-only CEF minimization across parameter proposals.

Matches EquilibriumModel's parameter and molar-atom conventions. No new physical
model or nonconvex support. chunk_size bounds temporary coefficient arrays to
O(chunk_size*nstate*5); returned arrays remain O(nparam*nstate). Default 2048.
Use smaller chunks when evaluating many states. No symbolic work per proposal.
"""
import numpy as np
try:
    from .equilibrium_solver import EquilibriumModel, NonConvexError, ConvergenceError, R
except ImportError:  # direct PYTHONPATH=prototype usage
    from equilibrium_solver import EquilibriumModel, NonConvexError, ConvergenceError, R


class BatchedEquilibriumModel:
    def __init__(self, x, T, chunk_size=2048):
        if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        self.model = EquilibriumModel(x, T)
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _poly(c, u):
        return ((c[..., 4]*u+c[..., 3])*u+c[..., 2])*u*u+c[..., 1]*u+c[..., 0]

    @staticmethod
    def _grad(c, u):
        return ((4*c[..., 4]*u+3*c[..., 3])*u+2*c[..., 2])*u+c[..., 1]

    @staticmethod
    def _curv(c, u):
        return (12*c[..., 4]*u+6*c[..., 3])*u+2*c[..., 2]

    def evaluate(self, theta, return_sites=False):
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 2 or theta.shape[1] != 10 or not np.all(np.isfinite(theta)):
            raise ValueError('Expected finite theta with shape (nparam,10)')
        m = self.model
        shape = (len(theta), len(m.x))
        out = {k: np.empty(shape) for k in ['energy', 'stationarity', 'convexity_lower_bound']}
        out['iterations'] = []
        if return_sites:
            out['sites'] = np.empty(shape+(2,))
        for start in range(0, len(theta), self.chunk_size):
            stop = min(start+self.chunk_size, len(theta))
            th = theta[start:stop]
            c = np.einsum('sjk,pj->psk', m.basis, th, optimize=False)
            low = np.minimum(self._curv(c, m.lo), self._curv(c, m.hi))
            den = 24*c[..., 4]
            vertex = np.divide(-6*c[..., 3], den, out=np.zeros_like(den), where=den != 0)
            valid = (c[..., 4] > 0) & (vertex > m.lo) & (vertex < m.hi)
            bound = np.where(valid, np.minimum(low, self._curv(c, vertex)), low)+m.entropy_bound
            bad = np.argwhere((bound <= 0) & m.interior)
            if len(bad):
                p, s = bad[0]
                raise NonConvexError(f'Convexity not certified: parameter row {start+p}, state {s}')
            u = np.broadcast_to(m.x, (len(th), len(m.x))).copy()
            residual = np.zeros_like(u)
            ids = np.flatnonzero(m.interior)
            iterations = 0
            if len(ids):
                cc = c[:, ids]
                bulk, t = m.x[ids], m.T[ids]
                lo = np.broadcast_to(m.lo[ids], (len(th), len(ids))).copy()
                hi = np.broadcast_to(m.hi[ids], lo.shape).copy()
                z = np.broadcast_to(bulk, lo.shape).copy()
                tol = 1e-8+1e-11*(R*t+np.max(np.abs(th[:, :9]), axis=1)[:, None]+np.abs(th[:, 9, None])*t)
                # Preserve converged cells while other proposals continue iterating.
                for iterations in range(100):
                    v = (4*bulk-z)/3
                    g = self._grad(cc, z)+R*t/4*(np.log(z)-np.log1p(-z)-np.log(v)+np.log1p(-v))
                    h = self._curv(cc, z)+R*t/4*(1/(z*(1-z))+1/(3*v*(1-v)))
                    done = np.abs(g) <= tol
                    if np.all(done):
                        break
                    lo = np.where((g < 0) & ~done, z, lo)
                    hi = np.where((g >= 0) & ~done, z, hi)
                    proposal = z-g/h
                    valid = (proposal > lo) & (proposal < hi) & np.isfinite(proposal)
                    z = np.where(done, z, np.where(valid, proposal, (lo+hi)/2))
                else:
                    raise ConvergenceError('Batched safeguarded Newton failed stationarity tolerance')
                u[:, ids] = z
                residual[:, ids] = g
            v = (4*m.x-u)/3
            if (np.any(u < 0) or np.any(u > 1) or np.any(v < 0) or np.any(v > 1)
                    or np.max(np.abs((u+3*v)/4-m.x), initial=0) > 1e-13):
                raise ConvergenceError('Batched site constraint failure')
            def mix(q):
                ans = np.zeros_like(q)
                mask = (q > 0) & (q < 1)
                z = q[mask]
                ans[mask] = z*np.log(z)+(1-z)*np.log1p(-z)
                return ans
            energy = self._poly(c, u)+R*m.T/4*(mix(u)+3*mix(v))
            if np.any(~np.isfinite(energy)):
                raise ConvergenceError('Nonfinite batched energy')
            out['energy'][start:stop] = energy
            out['stationarity'][start:stop] = residual
            out['convexity_lower_bound'][start:stop] = bound
            out['iterations'].append(iterations+1)
            if return_sites:
                out['sites'][start:stop, :, 0] = u
                out['sites'][start:stop, :, 1] = v
        return out
