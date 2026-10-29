# Project Structure and Technology Stack

## Technology stack

- **Language:** Python 3.11+
- **Packaging:** `pyproject.toml` with setuptools
- **Data:** pandas and NumPy
- **Machine learning:** scikit-learn, optional XGBoost, optional LightGBM
- **Validation:** pytest
- **Live API:** httpx and Pydantic settings/schemas
- **Streaming:** kafka-python
- **Configuration:** YAML plus environment variables loaded through
  `python-dotenv`
- **Artifacts:** CSV data, JSON metadata/reports, Markdown reports, and a
  joblib champion model

## Runtime flow

```text
Kafka ticks/news
      |
      v
market_data -> features -> strategy ensemble -> risk checks
      |                                      |
      +--------------------------------------v
                                  order manager / API client
                                                  |
                                                  v
                                      portfolio and monitoring
```

## Source map

| Layer | Location | Responsibility |
|---|---|---|
| Configuration | [`src/trading_bot/config.py`](src/trading_bot/config.py) | Environment-backed runtime settings |
| API | [`src/trading_bot/api/`](src/trading_bot/api/) | HTTP client and request/response schemas |
| Market data | [`src/trading_bot/market_data/`](src/trading_bot/market_data/) | Historical data, Kafka consumption, normalization, validation |
| Features | [`src/trading_bot/features/`](src/trading_bot/features/) | Technical indicators and news/advanced features |
| ML | [`src/trading_bot/ml/`](src/trading_bot/ml/) | Labels, datasets, candidate models, training, inference, stacking, calibration |
| Strategy | [`src/trading_bot/strategy/`](src/trading_bot/strategy/) | Regime detection, signals, and ensemble decisions |
| Risk | [`src/trading_bot/risk/`](src/trading_bot/risk/) | Exposure, limits, and position sizing |
| Execution | [`src/trading_bot/execution/`](src/trading_bot/execution/) | Order state management and safe reconciliation |
| Portfolio | [`src/trading_bot/portfolio/`](src/trading_bot/portfolio/) | Accounting and performance metrics |
| Monitoring | [`src/trading_bot/monitoring/`](src/trading_bot/monitoring/) | Drift and runtime monitoring |
| Research | [`src/trading_bot/research/`](src/trading_bot/research/) | Backtesting and time-series evaluation |
| Operations | [`scripts/`](scripts/) | Ingestion, training, backtests, and rebalancing commands |

## Important files

- [`scripts/train_ml.py`](scripts/train_ml.py): trains and evaluates the model suite.
- [`scripts/run_backtest.py`](scripts/run_backtest.py): runs the comparative
  strategy benchmark.
- [`src/trading_bot/main.py`](src/trading_bot/main.py): live bot entry point.
- [`config/risk_limits.yaml`](config/risk_limits.yaml): risk guardrails.
- [`config/settings.yaml`](config/settings.yaml): non-secret runtime defaults.
- [`tests/`](tests/): unit and integration coverage.

## Repository safety

`.env` is ignored and must contain local credentials only. `.env.example`
contains placeholders and documents the required variable names. Python caches,
test caches, logs, virtual environments, and unrelated generated artifacts are
also ignored. The small, explicitly named model artifacts are tracked because
they are part of the reproducibility record for this repository.
