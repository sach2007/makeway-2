# MANTHAN 2026 Institutional Event-Driven Backtest & Purged CV Report

## 1. Friction & Market Impact Specifications

- **Zero-Cost Impossibility**: Zero-cost results are hard-blocked by code assertions.
- **Brokerage Fee**: ₹20.00 flat per executed order.
- **Depository Participant (DP) Charge**: ₹15.34 flat per sell transaction.
- **Securities Transaction Tax (STT)**: 2.5 bps (0.025%) on all executed notionals.
- **Bid-Ask Spread**: 5 bps minimum half-spread.
- **Non-Linear Market Impact**: Square-root law $I = \gamma \sqrt{Q / V}$ with $\gamma = 0.10$.
- **Execution Latency Penalty**: 2 bps slippage simulating order transmission delay.

## 2. Full Period Strategy Performance (Baseline vs Ensemble)

| strategy              |    final_wealth |   total_return_pct |   cagr_pct |   sharpe_ratio |   sortino_ratio |   calmar_ratio |   max_drawdown_pct |   win_rate_pct |   profit_factor |   turnover_ratio |   total_frictions_paid |
|:----------------------|----------------:|-------------------:|-----------:|---------------:|----------------:|---------------:|-------------------:|---------------:|----------------:|-----------------:|-----------------------:|
| Baseline (Buy & Hold) |     2.15883e+06 |             115.88 |      32.51 |           1.14 |            1.15 |           1.87 |               17.4 |         100    |            1    |             0.7  |                  424.8 |
| Ensemble              | -1046.79        |            -100.1  |       0    |          -0.46 |           -0.91 |           0    |              100.1 |          17.54 |            0.65 |            79.81 |                15453.4 |

## 3. Purged & Embargoed Walk-Forward Validation (IS vs OOS Side-by-Side)

|   fold | train_period             | test_period              |   is_net_return_pct |   is_sharpe |   is_max_dd_pct |   oos_net_return_pct |   oos_sharpe |   oos_max_dd_pct |
|-------:|:-------------------------|:-------------------------|--------------------:|------------:|----------------:|---------------------:|-------------:|-----------------:|
|      1 | 2022-04-06 to 2023-06-22 | 2023-06-30 to 2023-12-28 |              -99.76 |       -7.38 |           99.74 |               -14.88 |        -2.3  |            18.95 |
|      2 | 2022-04-06 to 2023-12-20 | 2023-12-29 to 2024-07-01 |             -100.02 |       -0.91 |          100.02 |               -26.92 |        -2.93 |            27.1  |
|      3 | 2022-04-06 to 2024-06-24 | 2024-07-02 to 2024-12-30 |             -100.06 |       -0.59 |          100.06 |               -98.17 |        -8.59 |            98.15 |

## 4. Per-Regime Breakdown (Ensemble Strategy)

| regime          |   count |   mean_daily_return_pct |   annualized_sharpe |
|:----------------|--------:|------------------------:|--------------------:|
| BULL_TRENDING   |    2281 |                  -0.573 |               -0.66 |
| BEAR_TRENDING   |    1239 |                  -0.685 |               -0.65 |
| HIGH_VOLATILITY |     866 |                  -0.992 |               -1.07 |
| RANGE_BOUND     |    2238 |                  -0.046 |               -0.05 |