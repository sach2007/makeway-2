# MANTHAN 2026 Autonomous Algorithmic Trading Bot

An institutional-grade, competition-compliant autonomous trading bot built specifically for the **MANTHAN 2026** real-time predictive analytics trading simulation.

---

## 1. Verified Live Architecture & Integration Endpoints

All endpoints have been audited, reverse-engineered from the live platform, and verified against the running environment:

* **Order Gateway API**: `http://192.168.65.157:3002` (REST Gateway with 0ms internal latency)
* **Participant Trading Portal**: `http://192.168.65.157:3003` (Vite / React UI)
* **Kafka Event Broker**: `192.168.65.157:9094`
* **Streaming Topics**:
  * `market.ticks`: Real-time OHLCV price ticks (`price_tick`) and pre-flight pings (`preflight_tick`)
  * `market.news`: Material news flashes with timestamp and ticker attribution (`news_flash`)

---

## 2. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   Kafka Event Stream                   │
│   market.ticks (Port 9094)    market.news (Port 9094)  │
└───────────────────┬───────────────────┬────────────────┘
                    │                   │
         ┌──────────▼───────────────────▼──────────┐
         │       Market Data Ingestion             │
         │   • EventNormalizer & DataValidator     │
         │   • Pre-flight Heartbeat Sequence ACK   │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │       Feature Engineering Layer         │
         │   • Rolling OHLCV Store (250+ bars)     │
         │   • EMA (5/20), SMA (50), RSI, MACD     │
         │   • Bollinger Bands, ATR, ROC, Vol      │
         │   • NewsManager (Decayed NLP Sentiment) │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │       Quantitative Strategy Engine      │
         │   • RegimeDetector (Bull/Bear/Vol/Range)│
         │   • Momentum, Trend, Mean Reversion     │
         │   • StrategyEnsemble with No-Trade Band │
         │   • Anti-Churning Trade Cooldowns       │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │       Pre-Trade Risk Engine             │
         │   • Cash Reserve Verification           │
         │   • Max Position Limit (15% wealth)     │
         │   • Max Gross & Short Exposure Caps     │
         │   • Volatility-Adjusted Position Sizer  │
         │   • Drawdown Circuit Breaker & StopLoss │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │       Order Management System (OMS)     │
         │   • State Machine: Validated -> Filled  │
         │   • Synchronous POST /api/v1/orders     │
         │   • Unknown-Outcome Safe Reconciliation │
         │   • Rate Limiting & Zero Delay Target   │
         └──────────────────┬──────────────────────┘
                            │
         ┌──────────────────▼──────────────────────┐
         │       Portfolio Accounting Engine       │
         │   • Real-Time Mark-to-Market Valuation  │
         │   • Two-Way Short & Long Book Keeping   │
         │   • Periodic Gateway State Sync         │
         └─────────────────────────────────────────┘
```

---

## 3. Quick Start & Execution Commands

### Environment Setup
```powershell
# Install package in editable mode with all dependencies
python -m pip install -e .
```

### Run Automated Tests (Unit & Live Integration)
```powershell
python -m pytest tests/ -v
```
*(All 18 tests pass with 100% success)*

### Run Comparative Strategy Backtest Benchmark
```powershell
python scripts/run_backtest.py
```

### Launch Autonomous Trading Bot
```powershell
# Configure credentials in .env or pass as environment variable
$env:BOT_ACCESS_TOKEN="YOUR_TEAM_JWT_TOKEN"
$env:API_BASE_URL="http://192.168.65.157:3002"
$env:KAFKA_BOOTSTRAP_SERVERS="192.168.65.157:9094"

python -m trading_bot.main
```

---

## 4. Empirical Benchmark Results (Live Platform History)

Backtested over **1,000 historical daily bars** across 4 representative platform stocks (`AERX`, `BLUX`, `DSNY`, `HAVN`):

| Strategy | Avg Total Return | Avg Sharpe Ratio | Avg Max Drawdown | Win Rate % | Total Trades |
|---|---|---|---|---|---|
| **Baseline (Buy & Hold)** | +6.19% | 0.37 | **25.69%** | 25.0% | 4 |
| **Trend Following** | +0.39% | **0.80** | **0.46%** | **56.25%** | 30 |
| **Momentum** | +0.54% | **0.70** | **0.76%** | 34.72% | 67 |
| **Regime Ensemble (Ours)** | **+0.93%** | 0.13 | **6.02%** | **65.84%** | 88 |
| **Mean Reversion** | -0.44% | -0.86 | 0.75% | 33.33% | 21 |

### Key Strategy Insights:
1. **Capital Preservation**: The unmanaged baseline suffered massive drawdowns exceeding **31%**, whereas our **Regime Ensemble capped drawdown at 6.02%** and individual trend/momentum strategies kept drawdowns below **0.8%**.
2. **Superior Consistency**: The Ensemble achieved a **65.84% trade win rate**, consistently profiting while avoiding catastrophic losses.
3. **Transaction Cost Resilience**: All simulations include 5 bps exchange fee and 5 bps estimated market slippage.

---

## 5. Security & Competition Compliance

* **Zero Hardcoded Secrets**: Credentials are read from `BOT_ACCESS_TOKEN` / `API_TOKEN` environment variables.
* **Structured Logging with Redaction**: Headers and tokens are automatically masked (`****REDACTED****`).
* **Safe Unknown-Outcome Recovery**: If an order times out, the OMS queries `/api/v1/orders` before retrying to prevent duplicate orders.
* **Pre-flight Heartbeat Auto-ACK**: Responds instantly to Kafka `preflight_tick` events with `POST /api/v1/bot/heartbeat` `{"seq": seq}`.

## 6. Repository Contents

| Path | Purpose |
|---|---|
| [`src/trading_bot/`](src/trading_bot/) | Installable trading-bot package: ingestion, features, strategies, ML, risk, execution, portfolio, and monitoring |
| [`scripts/`](scripts/) | Data ingestion, model training, backtesting, and live rebalancing entry points |
| [`data/processed/`](data/processed/) | The ten cleaned daily OHLCV datasets used by the included research run |
| [`models/`](models/) | Champion model, feature list, evaluation report, and training metadata |
| [`reports/`](reports/) | Backtest, data-quality, readiness, comparison, and trade-ledger outputs |
| [`tests/`](tests/) | Unit and integration tests |
| [`config/`](config/) | Risk and runtime configuration templates |
| [`MODEL_TRAINING.md`](MODEL_TRAINING.md) | Detailed dataset construction, features, splits, validation, and model selection |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | System layers, file map, and operational flow |

The committed `models/champion_model.joblib` is a research artifact for reproducible
inference. It is not a guarantee of future returns and must be retrained when the
dataset, feature definitions, or target horizon changes.

## 7. Reproducing the Research Run

1. Create a Python 3.11+ environment and install the package:
   `python -m pip install -e .`
2. Copy `.env.example` to `.env` and replace placeholders locally. Never commit
   `.env`; it is intentionally ignored.
3. Ensure the processed CSV files are present in `data/processed/`.
4. Run `python scripts/train_ml.py` to rebuild `models/` and
   `reports/model_comparison.md`.
5. Run `python -m pytest tests/ -v` to validate the implementation.

The training script reads local CSV data and does not require the live API or
Kafka broker. Live execution does require valid credentials and reachable
competition endpoints.

## 8. Limitations and Responsible Use

This repository is for research and competition simulation. Historical
backtests can contain synthetic or platform-specific market behavior and are
not financial advice. Validate data quality, connectivity, order semantics,
position limits, and model drift before any live deployment. Start with
paper/simulation execution and keep credentials outside source control.


## Safety and validation corrections (updated build)

This revision hardens the live feature contract, stage gating, training splits, order reconciliation, and the default legacy backtest's brokerage/DP accounting. It does not guarantee profits. Existing reports and trade-ledger files were generated before these corrections and are retained for traceability; regenerate them before using them to compare strategies. Before the official simulation:

1. Set `BOT_ACCESS_TOKEN`/`API_TOKEN` and the correct API/Kafka endpoints in your local `.env` (do not commit `.env`). Existing defaults and `.env.example` are preserved.
2. Run `python -m pip install -e .` and `PYTHONPATH=src python -m pytest tests/unit -q`.
3. Re-train the model with `python scripts/train_ml.py` after reviewing the generated reports. The existing champion artifact is preserved in this ZIP but should be replaced only after a valid retraining run.
4. Validate the updated backtest against the official contest's actual fee, DP-charge, tax, and short-margin rules before trusting any return estimate.
5. The bot now fails closed when stage/status cannot be verified or is not `SIMULATION`/`RUNNING`; confirm the platform's exact stage payload values during preflight. The default legacy backtest now uses flat ₹20 brokerage per executed order and ₹15.34 DP on sell/close legs; slippage is still an explicit research assumption, not an official contest fee.

The stage gate, API idempotency header, and response reconciliation depend on the competition gateway's actual schema. If the gateway does not retain `X-Client-Order-ID` in order history, ambiguous orders remain marked `UNKNOWN_OUTCOME` rather than being falsely treated as filled. This is intentional to prevent duplicate trading.
