# Model Training and Data Specification

This document describes the training run represented by the committed files in
[`models/`](models/) and [`reports/`](reports/). The implementation of the
pipeline is in [`scripts/train_ml.py`](scripts/train_ml.py) and
[`src/trading_bot/ml/pipeline.py`](src/trading_bot/ml/pipeline.py).

## Dataset

The training input is ten cleaned daily OHLCV CSV files under
[`data/processed/`](data/processed/):

| Symbol | Bars |
|---|---:|
| AERX | 687 |
| BLUX | 686 |
| DSNY | 686 |
| HAVN | 686 |
| HMDP | 686 |
| MOTR | 686 |
| ORCX | 686 |
| PLNX | 686 |
| ROUT | 686 |
| SONA | 686 |
| **Total** | **6,861** |

Each file contains `date`, `open`, `high`, `low`, `close`, and `volume`.
Rows are sorted chronologically per symbol. The data inventory and quality
checks are recorded in [`reports/data_inventory.md`](reports/data_inventory.md)
and [`reports/data_quality.json`](reports/data_quality.json).

## Feature engineering

[`TimeSeriesFeaturePipeline`](src/trading_bot/ml/dataset.py) builds a strictly
causal feature matrix. It discards the first 50 warm-up bars and the final
three bars, because the target requires a future horizon of three trading days.
The 16 features are:

- 1-, 3-, 5-, and 10-day returns
- EMA(5)/EMA(20) and close/SMA(50) divergence
- normalized RSI(14) and MACD histogram ratio
- Bollinger %B and bandwidth
- ATR ratio and 10-/20-day rolling volatility
- volume ratio, high-low ratio, and close-open ratio

The binary target is `1` when the next three-day return exceeds the 20 bps
cost hurdle; otherwise it is `0`. Missing feature values are handled after
feature computation with forward fill, backward fill, and a final zero fill.

## Splits and validation

The combined multi-symbol dataset is sorted by date and split without shuffling:

- **65% training**
- **15% validation**
- **20% held-out test**

Candidate models are also evaluated with three expanding-window walk-forward
folds. Every fold trains only on observations earlier than its test window.
`StandardScaler` is fitted on each training slice only, preventing scaler
lookahead.

The validation split is retained for the pipeline's research layout; final
reported predictive and trading metrics are calculated on the held-out test
slice.

## Models

The candidate suite in [`src/trading_bot/ml/models.py`](src/trading_bot/ml/models.py)
contains:

1. XGBoost histogram classifier, configured for CUDA when available, with CPU
   fallback.
2. LightGBM histogram classifier.
3. Random Forest classifier.
4. L2-regularized Logistic Regression baseline.

Each model uses a common wrapper for scaling, probability prediction, and
accuracy, precision, recall, F1, ROC-AUC, and Brier-score evaluation.

## Selection and trading simulation

For each candidate, the pipeline combines walk-forward AUC with an
out-of-sample trading simulation. The selection score is:

```text
composite score = test Sharpe ratio + (2.0 * mean walk-forward AUC)
```

The simulation starts with simulated wealth of 1,000,000, allocates 20% of
wealth per position, enters at probability >= 0.53, exits below 0.48, and
charges 5 bps fee plus 5 bps slippage per side as implemented in the pipeline.
A buy-and-hold baseline is reported alongside the model candidates.

The committed run selected **LightGBM** as the champion and recorded:

- 4,115 training samples
- 949 validation samples
- 1,267 test samples
- 19.70% simulated net return
- 0.73 simulated Sharpe ratio
- 7.38% maximum drawdown
- 53.11% win rate

These are historical research results, not promises of future performance.
See [`reports/model_comparison.md`](reports/model_comparison.md) for the full
comparison and [`models/metadata.json`](models/metadata.json) for machine-readable
run metadata.

## Re-training

```powershell
python scripts/ingest_data.py
python scripts/train_ml.py
python -m pytest tests/ -v
```

Retraining overwrites the model and report artifacts. Review the generated
metrics, data-quality report, and feature names before using the model for
inference. Keep API credentials in a local `.env` file only.
