"""Pair spread calculations with separate formation and trading inputs."""

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
    design = np.column_stack((np.ones(explanatory.size), explanatory))
    intercept, beta = np.linalg.lstsq(design, dependent, rcond=None)[0]
    return float(intercept), float(beta)


def rolling_zscore(spread: np.ndarray, window: int = 30) -> np.ndarray:
    """At index t, compare current spread with the strictly prior window."""
    values = _finite_vector(spread, "spread")
    if window < 2:
        raise ValueError("window must be at least 2")
    zscore = np.zeros(values.size, dtype=float)
    for index in range(window, values.size):
        history = values[index - window:index]
        standard_deviation = history.std(ddof=1)
        if standard_deviation > 0:
            zscore[index] = (values[index] - history.mean()) / standard_deviation
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
    if entry <= 0 or exit < 0 or exit >= entry:
        raise ValueError("entry must be positive and exit must be in [0, entry)")
    positions = np.zeros(values.size, dtype=float)
    state = 0.0
    for index, value in enumerate(values):
        if state == 0.0:
            if value >= entry:
                state = -1.0
            elif value <= -entry:
                state = 1.0
        elif abs(value) <= exit:
            state = 0.0
        positions[index] = state
    return positions

