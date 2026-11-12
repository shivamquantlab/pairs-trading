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

