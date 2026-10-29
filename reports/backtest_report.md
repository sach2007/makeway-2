# MANTHAN 2026 Event-Driven Strategy Backtest Report

- **Execution Friction**: 5 bps brokerage fee + 5 bps slippage per side (10 bps round trip)
- **Depository Participant Charges**: ₹15.34 flat per sell execution
- **Risk Rules Enforced**: -5% Stop-Loss, +15% Take-Profit Target
- **Asset Universe**: 10 platform stocks (10 datasets)
- **Total Executed Trades in Ledger**: 894

### Strategy Average Performance Summary

| strategy              |   total_return_pct |   sharpe_ratio |   max_drawdown_pct |   win_rate_pct |   total_trades |
|:----------------------|-------------------:|---------------:|-------------------:|---------------:|---------------:|
| Baseline (Buy & Hold) |      165.498       |          0.818 |             30.177 |        100     |             10 |
| Ensemble              |        4.88701e+06 |          1.941 |              2.223 |         26.969 |            500 |
| Mean_reversion        |        5.484       |          0.676 |              2.76  |         43.377 |            305 |
| Momentum              |        5.318       |          0.437 |              5.441 |         34.935 |            496 |
| Trend                 |        4.637       |          0.517 |              3.763 |         35.697 |            455 |

### Per-Symbol Detailed Results

| ticker   | strategy              |     final_wealth |   total_return_pct |   sharpe_ratio |   max_drawdown_pct |   total_trades |   win_rate_pct |
|:---------|:----------------------|-----------------:|-------------------:|---------------:|-------------------:|---------------:|---------------:|
| AERX     | Baseline (Buy & Hold) |      1.08516e+06 |        8.52        |           0.26 |              21.15 |              1 |         100    |
| AERX     | Momentum              |      1.00332e+06 |        0.33        |           0.07 |               3.48 |             16 |          25    |
| AERX     | Trend                 |      1.01102e+06 |        1.1         |           0.31 |               2.43 |             14 |          28.57 |
| AERX     | Mean_reversion        |      1.01679e+06 |        1.68        |           0.36 |               1.92 |             19 |          30    |
| AERX     | Ensemble              |      6.33033e+07 |     6230.33        |           1.38 |               1.72 |             29 |          20    |
| BLUX     | Baseline (Buy & Hold) |      1.60182e+06 |       60.18        |           0.99 |              18.82 |              1 |         100    |
| BLUX     | Momentum              |      1.03217e+06 |        3.22        |           0.56 |               2.53 |             33 |          35.29 |
| BLUX     | Trend                 |      1.0403e+06  |        4.03        |           1.1  |               1.21 |             27 |          42.86 |
| BLUX     | Mean_reversion        |      1.05731e+06 |        5.73        |           1.03 |               2.23 |             17 |          55.56 |
| BLUX     | Ensemble              |      6.43164e+07 |     6331.64        |           1.52 |               2.35 |             29 |          26.67 |
| DSNY     | Baseline (Buy & Hold) |      1.03782e+06 |        3.78        |           0.19 |              34.95 |              1 |         100    |
| DSNY     | Momentum              | 972304           |       -2.77        |          -0.38 |               6.39 |             41 |          28.57 |
| DSNY     | Trend                 | 987067           |       -1.29        |          -0.19 |               5.9  |             35 |          33.33 |
| DSNY     | Mean_reversion        | 990861           |       -0.91        |          -0.1  |               4.15 |             35 |          27.78 |
| DSNY     | Ensemble              |      4.15283e+09 |   415183           |           2.16 |               2.32 |             45 |          26.09 |
| HAVN     | Baseline (Buy & Hold) |      1.12449e+06 |       12.45        |           0.32 |              33.34 |              1 |         100    |
| HAVN     | Momentum              | 962124           |       -3.79        |          -0.33 |               7.21 |             63 |          21.88 |
| HAVN     | Trend                 | 998034           |       -0.2         |          -0.01 |               3.96 |             49 |          24    |
| HAVN     | Mean_reversion        |      1.0044e+06  |        0.44        |           0.07 |               4.36 |             45 |          30.43 |
| HAVN     | Ensemble              |      1.33126e+11 |        1.33125e+07 |           2.51 |               2.18 |             61 |          25.81 |
| HMDP     | Baseline (Buy & Hold) |      1.31615e+06 |       31.62        |           0.58 |              17.83 |              1 |         100    |
| HMDP     | Momentum              |      1.02942e+06 |        2.94        |           0.48 |               3.84 |             27 |          42.86 |
| HMDP     | Trend                 |      1.0084e+06  |        0.84        |           0.18 |               2.43 |             25 |          30.77 |
| HMDP     | Mean_reversion        |      1.01744e+06 |        1.74        |           0.36 |               2.36 |             27 |          35.71 |
| HMDP     | Ensemble              |      1.26689e+08 |    12568.9         |           1.64 |               2.18 |             29 |          13.33 |
| MOTR     | Baseline (Buy & Hold) |      1.30976e+06 |       30.98        |           0.47 |              36.54 |              1 |         100    |
| MOTR     | Momentum              |      1.03313e+06 |        3.31        |           0.36 |               4.48 |             56 |          35.71 |
| MOTR     | Trend                 |      1.00851e+06 |        0.85        |           0.14 |               3.4  |             47 |          33.33 |
| MOTR     | Mean_reversion        |      1.05283e+06 |        5.28        |           0.7  |               3.39 |             49 |          40    |
| MOTR     | Ensemble              |      1.00372e+09 |   100272           |           1.86 |               2.53 |             57 |          27.59 |
| ORCX     | Baseline (Buy & Hold) |      2.22241e+06 |      122.24        |           1.15 |              22.14 |              1 |         100    |
| ORCX     | Momentum              |      1.09048e+06 |        9.05        |           0.96 |               3.67 |             40 |          45    |
| ORCX     | Trend                 |      1.07138e+06 |        7.14        |           0.99 |               2.52 |             34 |          47.06 |
| ORCX     | Mean_reversion        |      1.07626e+06 |        7.63        |           0.92 |               2.46 |             27 |          42.86 |
| ORCX     | Ensemble              |      3.30288e+07 |     3202.88        |           1.25 |               1.97 |             39 |          35    |
| PLNX     | Baseline (Buy & Hold) |      9.83861e+06 |      883.86        |           1.65 |              46.05 |              1 |         100    |
| PLNX     | Momentum              |      1.16746e+06 |       16.75        |           0.8  |              11    |            111 |          35.71 |
| PLNX     | Trend                 |      1.12954e+06 |       12.95        |           0.78 |               6.96 |            113 |          36.84 |
| PLNX     | Mean_reversion        |      1.19673e+06 |       19.67        |           1.6  |               1.3  |             28 |          71.43 |
| PLNX     | Ensemble              |      2.80554e+11 |        2.80553e+07 |           2.67 |               1.78 |             87 |          34.09 |
| ROUT     | Baseline (Buy & Hold) |      2.5828e+06  |      158.28        |           1.03 |              30.15 |              1 |         100    |
| ROUT     | Momentum              |      1.08564e+06 |        8.56        |           0.66 |               4.42 |             50 |          36    |
| ROUT     | Trend                 |      1.06342e+06 |        6.34        |           0.63 |               3.6  |             48 |          33.33 |
| ROUT     | Mean_reversion        |      1.06571e+06 |        6.57        |           0.87 |               2.48 |             35 |          50    |
| ROUT     | Ensemble              |      6.84804e+10 |        6.84794e+06 |           2.43 |               2.43 |             71 |          27.78 |
| SONA     | Baseline (Buy & Hold) |      4.43069e+06 |      343.07        |           1.54 |              40.8  |              1 |         100    |
| SONA     | Momentum              |      1.15585e+06 |       15.58        |           1.19 |               7.39 |             59 |          43.33 |
| SONA     | Trend                 |      1.14611e+06 |       14.61        |           1.24 |               5.22 |             63 |          46.88 |
| SONA     | Mean_reversion        |      1.07005e+06 |        7.01        |           0.95 |               2.95 |             23 |          50    |
| SONA     | Ensemble              |      1.10621e+09 |   110521           |           1.99 |               2.77 |             53 |          33.33 |

### Trade Ledger Sample (Latest 10 Trades)

| trade_id      | ticker   | strategy   | action     | entry_date   | exit_date   |   shares |   entry_price |   exit_price |           net_pnl |   return_pct | exit_reason          |
|:--------------|:---------|:-----------|:-----------|:-------------|:------------|---------:|--------------:|-------------:|------------------:|-------------:|:---------------------|
| TRD_SONA_0018 | SONA     | ensemble   | SELL       | 2023-10-26   | 2023-11-27  |   358667 |        158.46 |       182.95 |       8.75068e+06 |        15.4  | TAKE_PROFIT (+15.5%) |
| TRD_SONA_0019 | SONA     | ensemble   | SELL       | 2023-11-29   | 2024-01-24  |   290049 |        182.4  |       211.03 |       8.27443e+06 |        15.64 | TAKE_PROFIT (+15.8%) |
| TRD_SONA_0020 | SONA     | ensemble   | SELL       | 2024-01-26   | 2024-02-22  |   261312 |        214.4  |       248    |       8.74735e+06 |        15.61 | TAKE_PROFIT (+15.7%) |
| TRD_SONA_0021 | SONA     | ensemble   | SELL       | 2024-02-26   | 2024-04-04  |   240389 |        255.12 |       295.81 |       9.74695e+06 |        15.89 | TAKE_PROFIT (+16.0%) |
| TRD_SONA_0022 | SONA     | ensemble   | SELL       | 2024-04-08   | 2024-04-15  |   220341 |        309.22 |       290.37 |      -4.18539e+06 |        -6.14 | STOP_LOSS (-6.0%)    |
| TRD_SONA_0023 | SONA     | ensemble   | SELL       | 2024-04-17   | 2024-04-19  |   184514 |        293.73 |       275.69 |      -3.35311e+06 |        -6.19 | STOP_LOSS (-6.1%)    |
| TRD_SONA_0024 | SONA     | ensemble   | SELL       | 2024-05-20   | 2024-07-18  |   140163 |        308.36 |       290.01 |      -2.59222e+06 |        -6    | STOP_LOSS (-5.9%)    |
| TRD_SONA_0025 | SONA     | ensemble   | COVER      | 2024-07-24   | 2024-09-19  |    95909 |        336.35 |       360.4  |      -2.32375e+06 |        -7.2  | STOP_LOSS (-7.1%)    |
| TRD_SONA_0026 | SONA     | ensemble   | SELL       | 2024-09-23   | 2024-11-13  |   345489 |        369.19 |       467.14 |       3.37571e+07 |        26.47 | TAKE_PROFIT (+26.6%) |
| TRD_SONA_0027 | SONA     | ensemble   | SELL_FINAL | 2024-11-15   | 2025-01-02  |   302077 |        458.55 |       457.56 | -367593           |        -0.27 | SIMULATION_END       |