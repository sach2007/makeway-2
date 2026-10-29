# Updated source audit — 2026-10-30

This file records source-level corrections made to the uploaded ZIP. It is not a claim that the competition gateway or live simulation was tested.

## Changes

- Live `FeatureExtractor` now emits the exact 16 named model features used by `TimeSeriesFeaturePipeline`. Inference refuses missing feature keys and disables a model whose persisted feature order differs from the training schema, instead of silently substituting zeroes.
- Dataset cleaning no longer backward-fills missing features from future observations. Chronological splits are grouped by distinct dates, with a configurable purged boundary based on the forward-label horizon. Walk-forward CV uses the same date-grouped, purged splitter.
- ML trade proxy tracks positions per ticker instead of using one shared holding flag across all symbols. It applies flat ₹20 entry/exit brokerage and ₹15.34 delivery-sale DP assumptions. It remains a research proxy, not a full portfolio/OMS replay.
- Main live loop refreshes stage/status at a bounded cadence and fails closed unless stage is `SIMULATION`/`SIM` and status is `RUNNING`. It also syncs portfolio state independently of early-return strategy branches and passes current ATR context to exit checks.
- OMS sends a stable client order ID and will not call a possibly old order a fill based only on matching ticker/action/quantity. If gateway order history does not expose the client ID, the state remains `UNKNOWN_OUTCOME` pending authoritative sync.
- Legacy backtest uses flat ₹20 brokerage per executed order and applies the ₹15.34 DP charge only to delivery sale legs; a short-close cash double-counting error was corrected. Slippage remains a research assumption and taxes/other platform-specific costs must be checked against the official rules.
- Added unit tests for live/training feature parity, finite causal training features, and date-grouped splits.

## Verification

- Unit tests: `PYTHONPATH=src python -m pytest tests/unit -q` — 52 passed in the packaged build environment.
- Not run: live API orders, Kafka connectivity, authenticated integration tests, or official competition simulation. Those require the actual venue network/token and must be tested by the team.
- Retrain with `python scripts/train_ml.py` and regenerate all backtests/reports before relying on performance metrics. The existing model artifact was preserved, not silently replaced.

## Important platform dependency

The live stage gate assumes the stage response identifies simulation with `stage=SIMULATION` (or `SIM`) and active trading with `status=RUNNING`. Confirm this against the official API response during preflight. If the API uses different values, update the allow-list only after verifying the official contract; do not disable the gate. The OMS idempotency improvement depends on the gateway persisting `X-Client-Order-ID` in order history. If not, ambiguous orders intentionally remain unknown rather than being blindly retried.
