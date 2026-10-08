"""Full-support posterior-focused quadrature for a bounded two-dimensional density."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp


def focused_axis(lower, upper, center, scale, count):
    if not np.isfinite([lower, upper, center, scale]).all() or lower >= upper or scale <= 0:
        raise ValueError("Finite ordered bounds and a positive scale required")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 3:
        raise ValueError("At least three axis nodes required")
    coordinate = np.linspace(np.arcsinh((lower-center)/scale), np.arcsinh((upper-center)/scale), count)
    axis = center+scale*np.sinh(coordinate)
    axis[[0, -1]] = [lower, upper]
    if np.any(np.diff(axis) <= 0):
        raise ValueError("Axis cannot be resolved in floating-point arithmetic")
    return axis


def trapezoid_weights(axis):
    widths = np.diff(axis)
    if len(widths) < 1 or np.any(widths <= 0):
        raise ValueError("Strictly increasing axis required")
    return np.r_[widths[0]/2, (widths[:-1]+widths[1:])/2, widths[-1]/2]


def transformed_simpson_weights(axis, center, scale):
    if len(axis) < 3 or len(axis) % 2 != 1:
        raise ValueError("An odd number of nodes is required for Simpson integration")
    coordinate = np.arcsinh((axis-center)/scale)
    spacing = (coordinate[-1]-coordinate[0])/(len(axis)-1)
    if not np.allclose(np.diff(coordinate), spacing, rtol=1e-10, atol=1e-12):
        raise ValueError("Uniform transformed coordinates required")
    factors = np.ones(len(axis))
    factors[1:-1:2] = 4
    factors[2:-1:2] = 2
    return factors*spacing/3*np.sqrt(scale**2+(axis-center)**2)


def marginal_probabilities(axis, density, value, probabilities=(.025, .975)):
    """Integrate a piecewise-linear marginal and invert its quadratic cell CDF."""
    axis, density = np.asarray(axis, float), np.asarray(density, float)
    if axis.shape != density.shape or not np.isfinite(density).all() or np.any(density < 0):
        raise ValueError("Finite nonnegative marginal density required")
    trapezoid_weights(axis)
    cumulative = cumulative_trapezoid(density, axis, initial=0.)
    if cumulative[-1] <= 0:
        raise ValueError("Positive marginal integral required")
    density = density/cumulative[-1]
    cumulative = cumulative/cumulative[-1]
    slopes = np.diff(density)/np.diff(axis)
    if value <= axis[0]:
        probability = 0.
    elif value >= axis[-1]:
        probability = 1.
    else:
        cell = np.searchsorted(axis, value, side="right")-1
        offset = value-axis[cell]
        probability = cumulative[cell]+density[cell]*offset+.5*slopes[cell]*offset**2
    quantiles = []
    for target in probabilities:
        if not 0 < target < 1:
            raise ValueError("Quantile probabilities must lie strictly inside (0,1)")
        cell = min(np.searchsorted(cumulative, target, side="right")-1, len(axis)-2)
        residual = target-cumulative[cell]
        discriminant = max(0., density[cell]**2+2*slopes[cell]*residual)
        denominator = density[cell]+np.sqrt(discriminant)
        if denominator <= 0:
            raise ValueError("Unresolved quantile cell")
        offset = 2*residual/denominator
        quantiles.append(float(axis[cell]+np.clip(offset, 0., axis[cell+1]-axis[cell])))
    return float(np.clip(probability, 0., 1.)), quantiles


def integrate(log_density, bounds, center, scales, values, count):
    """Simpson integration includes the sinh Jacobian over the complete support."""
    bounds = np.asarray(bounds, float)
    center, scales, values = map(lambda vector: np.asarray(vector, float), (center, scales, values))
    if bounds.shape != (2, 2) or any(vector.shape != (2,) for vector in (center, scales, values)):
        raise ValueError("Exactly two coefficient dimensions required")
    axes = [focused_axis(*bounds[index], center[index], scales[index], count) for index in range(2)]
    weights = [transformed_simpson_weights(axis, center[index], scales[index]) for index, axis in enumerate(axes)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([coordinate.ravel() for coordinate in mesh])
    logarithms = np.asarray(log_density(points), float).reshape(count, count)
    if np.any(np.isnan(logarithms)) or np.any(np.isposinf(logarithms)) or not np.isfinite(logarithms).any():
        raise ValueError("Invalid log density")
    log_mass = logarithms+np.log(weights[0][:, None])+np.log(weights[1][None, :])
    log_integral = float(logsumexp(log_mass))
    joint_mass = np.exp(log_mass-log_integral)
    result = dict(mean=[], sd=[], cdfs=[], quantiles=[], log_integral=log_integral,
                  nodes_per_axis=count, evaluations=count**2)
    for index in range(2):
        marginal_mass = joint_mass.sum(axis=1-index)
        mean = float(np.sum(marginal_mass*axes[index]))
        variance = float(np.sum(marginal_mass*(axes[index]-mean)**2))
        probability, quantiles = marginal_probabilities(axes[index], marginal_mass/weights[index], values[index])
        result["mean"].append(mean)
        result["sd"].append(float(np.sqrt(variance)))
        result["cdfs"].append(probability)
        result["quantiles"].append(quantiles)
    return result
