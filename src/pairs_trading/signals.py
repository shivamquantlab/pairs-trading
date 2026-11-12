"""Pair spread calculations with separate formation and trading inputs."""

from math import frexp, ldexp

import numpy as np


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector with at least 2 values")
    return result


def estimate_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Estimate intercept and beta from the caller-supplied formation sample."""
    dependent = _finite_vector(y, "y")
    explanatory = _finite_vector(x, "x")
    if dependent.size != explanatory.size:
        raise ValueError("y and x must have the same length")
    x_scale = float(np.max(np.abs(explanatory)))
    if x_scale == 0.0:
        raise ValueError("x must have non-zero variation")
    scaled_x = explanatory / x_scale
    centered_x = scaled_x - scaled_x.mean()
    centered_scale = float(np.max(np.abs(centered_x)))
    if centered_scale == 0.0:
        raise ValueError("x must have non-zero variation")
    centered_x /= centered_scale
    x_mean = x_scale * float(scaled_x.mean())

    y_scale = float(np.max(np.abs(dependent)))
    if y_scale == 0.0:
        return 0.0, 0.0
    scaled_y = dependent / y_scale
    y_mean = y_scale * float(scaled_y.mean())
    centered_y = scaled_y - scaled_y.mean()
    beta_scaled = float(centered_x @ centered_y / (centered_x @ centered_x))
    if beta_scaled == 0.0:
        beta = 0.0
    else:
        beta_mantissa, beta_exponent = frexp(beta_scaled)
        y_mantissa, y_exponent = frexp(y_scale)
        x_mantissa, x_exponent = frexp(x_scale)
        centered_mantissa, centered_exponent = frexp(centered_scale)
        slope_mantissa = beta_mantissa * y_mantissa / (x_mantissa * centered_mantissa)
        try:
            beta = ldexp(
                slope_mantissa,
                beta_exponent + y_exponent - x_exponent - centered_exponent,
            )
        except OverflowError as error:
            raise ValueError("formation hedge ratio exceeds the finite float range") from error
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        intercept = y_mean - beta * x_mean
    if not np.isfinite([intercept, beta]).all():
        raise ValueError("formation regression coefficients exceed the finite float range")
    return float(intercept), float(beta)


def rolling_zscore(spread: np.ndarray, window: int = 30) -> np.ndarray:
    """At index t, compare current spread with the strictly prior window."""
    values = _finite_vector(spread, "spread")
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 2
    ):
        raise ValueError("window must be an integer of at least 2")
    zscore = np.zeros(values.size, dtype=float)
    for index in range(window, values.size):
        history = values[index - window:index]
        scale = float(max(np.max(np.abs(history)), abs(values[index])))
        if scale == 0.0:
            continue
        scaled_history = history / scale
        standard_deviation = scaled_history.std(ddof=1)
        if standard_deviation > 0:
            zscore[index] = (values[index] / scale - scaled_history.mean()) / standard_deviation
            if not np.isfinite(zscore[index]):
                raise ValueError("z-score exceeds the finite float range")
    return zscore


def pair_positions(
    zscore_at_close: np.ndarray,
    *,
    entry: float = 2.0,
    exit: float = 0.5,
) -> np.ndarray:
    """Use close-t z-score to choose a spread position for the next interval."""
    values = np.asarray(zscore_at_close, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("zscore_at_close must be a finite vector")
    try:
        entry_level = float(entry)
        exit_level = float(exit)
    except (TypeError, ValueError) as error:
        raise ValueError("entry and exit thresholds must be finite numbers") from error
    if (
        not np.isfinite([entry_level, exit_level]).all()
        or entry_level <= 0
        or exit_level < 0
        or exit_level >= entry_level
    ):
        raise ValueError("entry must be positive and exit must be in [0, entry)")
    positions = np.zeros(values.size, dtype=float)
    state = 0.0
    for index, value in enumerate(values):
        if state == 0.0:
            if value >= entry_level:
                state = -1.0
            elif value <= -entry_level:
                state = 1.0
        elif abs(value) <= exit_level:
            state = 0.0
        positions[index] = state
    return positions

