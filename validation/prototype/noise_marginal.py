"""Exact scalar-noise marginalization for iid Gaussian residuals.

Likelihood: (2*pi)^(-n/2) sigma^(-n) exp(-RSS/(2 sigma^2)).
Prior: HalfNormal(s0). For w=sigma^2 and p=(1-n)/2,
I = integral sigma^(-n) exp(-RSS/(2sigma^2)-sigma^2/(2s0^2)) d_sigma
  = (RSS*s0^2)^(p/2) K_p(sqrt(RSS)/s0).
Conditional w ~ scipy.stats.geninvgauss(p, sqrt(RSS)/s0,
                                      scale=sqrt(RSS)*s0).
No finite noise truncation is imposed. Extremely ill-scaled Bessel arguments
fail explicitly rather than silently producing invalid probabilities.
Sources: scipy.special.kve and scipy.stats.geninvgauss official docs.
"""
import numbers
import numpy as np
from scipy.special import kve
from scipy.stats import geninvgauss


def _inputs(rss,n,s0):
    if isinstance(n,(bool,np.bool_)) or not isinstance(n,numbers.Integral) or n<1:
        raise ValueError('n must be a positive integer observation count')
    if np.ndim(s0)!=0 or not np.isfinite(s0) or s0<=0:
        raise ValueError('s0 must be finite and positive')
    r=np.asarray(rss,dtype=float)
    if r.size==0 or np.any(~np.isfinite(r)) or np.any(r<0):
        raise ValueError('RSS must be nonempty, finite and nonnegative')
    if np.any(r==0):
        raise ValueError('RSS=0 gives a divergent sigma integral at sigma=0 for n>=1; this is not a finite likelihood')
    p=(1.-int(n))/2
    log_scale=.5*np.log(r)+np.log(s0)
    log_z=.5*np.log(r)-np.log(s0)
    with np.errstate(over='raise',under='ignore',invalid='raise'):
        try:z=np.exp(log_z)
        except FloatingPointError as exc:raise ValueError('Noise scales outside numerical range') from exc
    if np.any(~np.isfinite(z)) or np.any(z<=0):raise ValueError('Bessel argument outside numerical range')
    return r,p,log_scale,z


def _log_k(order,z):
    val=kve(abs(order),z)
    if np.any(~np.isfinite(val)) or np.any(val<=0):
        raise ValueError('Scaled Bessel evaluation outside supported numerical range; rescale or use higher precision')
    return np.log(val)-z


def log_marginal(rss,n,s0=50.,include_constants=False):
    """Scalar/array log integral. Set include_constants for normalized likelihood."""
    _,p,log_scale,z=_inputs(rss,n,s0)
    answer=p*log_scale+_log_k(p,z)
    if include_constants:
        answer+=.5*np.log(2/np.pi)-np.log(s0)-n/2*np.log(2*np.pi)
    if not np.isfinite(answer).all():raise ValueError('Nonfinite log marginal')
    return answer


def conditional_moment(rss,n,s0=50.,power=1.):
    """E[sigma**power | RSS], including noninteger and negative moments."""
    _,p,log_scale,z=_inputs(rss,n,s0)
    if np.ndim(power)!=0 or not np.isfinite(power):raise ValueError('Moment power must be finite scalar')
    # Same z makes the exponential scaling cancel exactly.
    a=kve(abs(p+power/2),z);b=kve(abs(p),z)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a<=0) or np.any(b<=0):
        raise ValueError('Moment Bessel evaluation outside supported numerical range')
    with np.errstate(over='raise',invalid='raise'):
        try:answer=np.exp(power/2*log_scale+np.log(a)-np.log(b))
        except FloatingPointError as exc:raise ValueError('Conditional moment outside numerical range') from exc
    if np.any(~np.isfinite(answer)) or np.any(answer<=0):raise ValueError('Conditional moment outside numerical range')
    return answer


def draw_sigmas(rss,n,s0=50.,rng=None,size=None):
    """Joint-posterior step: each RSS receives its own conditional sigma draw.

    Vector RSS defaults to one draw per entry; scalar RSS can specify size.
    A Generator or integer seed can be passed as rng.
    """
    _,p,log_scale,z=_inputs(rss,n,s0)
    if size is not None:
        dims=(size,) if np.isscalar(size) else tuple(size)
        if any(isinstance(k,(bool,np.bool_)) or not isinstance(k,numbers.Integral) or k<1 for k in dims):
            raise ValueError('size must contain positive integers')
    # Draw dimensionless GIG, then apply scale in logs to avoid squaring sigma.
    samples=geninvgauss.rvs(p,z,size=size,random_state=np.random.default_rng(rng))
    if np.any(~np.isfinite(samples)) or np.any(samples<=0):raise ValueError('Invalid conditional GIG draw')
    with np.errstate(over='raise',invalid='raise'):
        try:answer=np.exp(.5*(np.log(samples)+log_scale))
        except FloatingPointError as exc:raise ValueError('Conditional draw outside numerical range') from exc
    if np.any(~np.isfinite(answer)) or np.any(answer<=0):raise ValueError('Invalid conditional sigma draw')
    return answer
