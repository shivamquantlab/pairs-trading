# Pairs Trading

## Research question
Can a formation-only linear hedge ratio and a past-window spread z-score be translated into a signal without fitting on the trading period?

## Methodology and formulas
The formation window estimates log(y_t)=alpha+beta*log(x_t)+epsilon_t by least squares. The beta is frozen for the test period. At time t, the spread z-score compares the current spread with its strictly prior W observations. A state machine enters short spread above the positive entry threshold, long below the negative entry threshold, and exits near zero. The close-t state is applied to the following spread change.

## Assumptions and limitations
- The demo creates a shared synthetic log-price component plus a mean-reverting synthetic residual with seed 101. There are no real securities or market prices.
- The strategy is a two-leg unit-exposure illustration; cumulative PnL is in synthetic log-spread units, not portfolio returns.
- It charges an illustrative 3 bp per leg per turnover unit. Borrow availability/cost, execution delay, corporate actions, delistings, and survivorship bias are not modeled.
- Cointegration is not formally tested; selecting pairs, thresholds, and windows requires a separate multiple-testing-aware research design.
- Results are not investment performance.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and pairs-demo --seed 101. The experiment estimates beta only on the first 150 observations and writes a synthetic chart and JSON summary. No APIs or secrets are used.

