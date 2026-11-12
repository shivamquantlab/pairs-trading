import numpy as np
import pytest

from pairs_trading.signals import estimate_hedge_ratio, pair_positions, rolling_zscore


def test_formation_sample_recovers_hedge_ratio():
    x = np.arange(5.0)
    y = 3.0 + 1.5 * x
    intercept, beta = estimate_hedge_ratio(y, x)
    assert intercept == pytest.approx(3.0)
    assert beta == pytest.approx(1.5)


def test_rolling_zscore_uses_only_prior_window():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    altered_future = values.copy()
    altered_future[5] = 1000.0
    first = rolling_zscore(values, window=4)
    second = rolling_zscore(altered_future, window=4)
    assert first[4] == pytest.approx(second[4])
    assert first[4] > 0


def test_pair_state_enters_and_exits_with_hysteresis():
    positions = pair_positions(np.array([0.0, 2.5, 1.0, 0.2, -2.5]), entry=2.0, exit=0.5)
    np.testing.assert_array_equal(positions, [0.0, -1.0, -1.0, 0.0, 1.0])


def test_constant_formation_regressor_is_rejected():
    with pytest.raises(ValueError, match="non-zero variation"):
        estimate_hedge_ratio(np.array([1.0, 2.0, 3.0]), np.ones(3))


def test_noninteger_window_and_nonfinite_thresholds_are_rejected():
    with pytest.raises(ValueError, match="integer"):
        rolling_zscore(np.arange(5.0), window=2.5)
    with pytest.raises(ValueError, match="entry must be positive"):
        pair_positions(np.array([0.0, 3.0]), entry=np.nan)


def test_hedge_ratio_is_stable_for_large_finite_log_values():
    x = np.array([-2e150, -1e150, 1e150, 2e150])
    y = 3e150 + 1.5 * x
    intercept, beta = estimate_hedge_ratio(y, x)
    assert beta == pytest.approx(1.5)
    assert intercept == pytest.approx(3e150, rel=1e-12)


def test_hedge_ratio_avoids_intermediate_overflow_for_large_values():
    x = np.array([-1e308, -1e308, 1e308])
    y = 0.5 * x
    intercept, beta = estimate_hedge_ratio(y, x)
    assert intercept == pytest.approx(0.0, abs=1e292)
    assert beta == pytest.approx(0.5)

