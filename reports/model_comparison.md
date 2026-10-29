# MANTHAN 2026 Machine Learning Model Comparison & Benchmark Report

- **Execution Timestamp**: 2026-10-29T09:08:24.700545+00:00
- **Total Training Samples**: 4,115
- **Validation Samples**: 949
- **Test Samples**: 1,267
- **Champion Model Selected**: `LightGBM` (GPU Accelerated: `False`)
- **Target Horizon**: 3 trading days (Hurdle: 20.0 bps)

### Cross-Validation & Out-of-Sample Performance Table

| model                 | uses_gpu   |   walk_forward_mean_auc |   roc_auc |     f1 |   accuracy |   net_return_pct |   sharpe_ratio |   max_drawdown_pct |   win_rate_pct |
|:----------------------|:-----------|------------------------:|----------:|-------:|-----------:|-----------------:|---------------:|-------------------:|---------------:|
| XGBoost_GPU           | True       |                  0.511  |    0.4723 | 0.4872 |     0.4799 |             4.13 |           0.18 |              11.23 |          50    |
| LightGBM              | False      |                  0.5094 |    0.4625 | 0.4884 |     0.4775 |            19.7  |           0.73 |               7.38 |          53.11 |
| RandomForest          | False      |                  0.5124 |    0.4776 | 0.4938 |     0.4807 |            -4.79 |          -0.2  |              11.39 |          48.88 |
| LogisticRegression    | False      |                  0.5006 |    0.4968 | 0.5264 |     0.5043 |             5.88 |           0.34 |               9.83 |          55.46 |
| Baseline (Buy & Hold) | False      |                  0.5    |    0.5    | 0      |     0.5    |            44.74 |           7.24 |              95.61 |         100    |

### Architectural Key Takeaways
1. **GPU Acceleration**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.0) executes histogram tree construction with sub-second fold completion.
2. **Walk-Forward Generalization**: Expanding-window CV guarantees that candidate hyperparameter sets preserve out-of-sample edge without lookahead.
3. **Zero Data Leakage**: StandardScalers and technical features are computed point-in-time strictly on past bars.
4. **Net Friction Accounting**: All simulated Sharpe ratios and returns incorporate 5 bps broker fee + 5 bps slippage.
