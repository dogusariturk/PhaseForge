"""Independent Gaussian-coefficient regression with scalar noise quadrature.

y | beta,sigma ~ N(X beta, sigma^2 I); beta ~ N(0,diag(tau^2));
sigma ~ HalfNormal(noise_scale). Units must be consistent (here eV/atom).
The coefficients are marginalized analytically. Quadrature is in log(sigma),
including its Jacobian. Joint draws use the conditional Gaussian distribution.
"""
import itertools
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.optimize import brentq

ELEMENTS = ['HF', 'NB', 'TI', 'V']
PAIRS = list(itertools.combinations(range(4), 2))
TRIPLES = list(itertools.combinations(range(4), 3))
LABELS = [f'{ELEMENTS[i]}-{ELEMENTS[j]}:L{k}' for i, j in PAIRS for k in (0, 1)] + [f'{"-".join(ELEMENTS[i] for i in ids)}:L0' for ids in TRIPLES]


def mm(a, b):
    """Small vector/matrix contraction without platform BLAS flag artifacts."""
    a, b = np.asarray(a), np.asarray(b)
    return np.einsum('...i,i->...', a, b, optimize=False) if b.ndim == 1 else np.einsum('...i,ij->...j', a, b, optimize=False)


def basis(x):
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[1] != 4 or not np.isfinite(x).all() or np.any(x < 0) or not np.allclose(x.sum(1), 1, atol=1e-12, rtol=0):
        raise ValueError('Expected finite normalized four-component compositions')
    return np.column_stack([x[:, i]*x[:, j]*(x[:, i]-x[:, j])**k for i, j in PAIRS for k in (0, 1)] + [np.prod(x[:, ids], axis=1) for ids in TRIPLES])



from phaseforge_uq._linear import Posterior, mm
