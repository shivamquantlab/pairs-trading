"""Seeded synthetic cointegrated pair; beta is frozen after formation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .signals import estimate_hedge_ratio, pair_positions, rolling_zscore


def run(output_dir: Path, seed: int = 101) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observations, formation = 450, 150
    common = np.cumsum(rng.normal(0.0, 0.01, observations))
    residual = np.zeros(observations)
    innovations = rng.normal(0.0, 0.006, observations)
    for index in range(1, observations):
        residual[index] = 0.92 * residual[index - 1] + innovations[index]
    log_x = 4.0 + common
    log_y = 0.3 + 1.25 * common + residual
    intercept, beta = estimate_hedge_ratio(log_y[:formation], log_x[:formation])
    test_spread = log_y[formation:] - intercept - beta * log_x[formation:]
    zscore = rolling_zscore(test_spread, window=30)
    positions = pair_positions(zscore[:-1], entry=1.5, exit=0.4)
    spread_change = np.diff(test_spread)
    turnover = np.abs(np.diff(np.concatenate(([0.0], positions))))
    # Two synthetic legs; cost is a labeled illustrative 3 bp per leg and turnover unit.
    net_pnl = positions * spread_change - turnover * (2.0 * 3.0 / 10_000.0)
    cumulative_pnl = np.cumsum(net_pnl)
    summary = {
        "seed": seed,
        "formation_observations": formation,
        "trading_observations": int(test_spread.size),
        "formation_intercept": intercept,
        "formation_hedge_ratio": beta,
        "two_leg_cost_bps_per_turnover_unit": 6.0,
        "synthetic_cumulative_spread_pnl": float(cumulative_pnl[-1]),
        "position_changes": float(turnover.sum()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(test_spread, color="#1d4ed8")
    axes[0].set_ylabel("Synthetic log spread")
    axes[1].plot(cumulative_pnl, color="#b91c1c")
    axes[1].set(xlabel="Synthetic trading observation", ylabel="Cumulative spread PnL")
    fig.suptitle("Pairs signals use formation-only hedge ratio")
    fig.text(0.5, 0.01, "SYNTHETIC • illustrative spread units, not investment performance",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_dir / "synthetic_pair.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic pair report to {args.output_dir}")


if __name__ == "__main__":
    main()

