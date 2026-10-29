# Quantitative System Audit & Architecture Review (Phase 0)

**Project**: MANTHAN 2026 Autonomous ML Trading System  
**Auditor**: Senior Quantitative Research & Systems Engineer  
**Scope**: Full Repository Audit (`src/trading_bot/`, `config/`, `scripts/`, `tests/`)  
**Timestamp**: 2026-10-29  

---

## 1. Executive Summary

A comprehensive audit of the existing autonomous trading bot codebase was conducted to identify algorithmic bugs, look-ahead leaks, mathematical flaws, and execution risks before implementing system upgrades. 

While the fundamental modular infrastructure (Kafka ingestion, REST Order Gateway, Pydantic configuration, structured logging) is functional and verified against live platform endpoints, several critical quantitative and architectural flaws exist in the strategy, feature engineering, labeling, risk management, and backtesting layers.

---

## 2. Detailed Findings & Bug Inventory

### Finding 1: Signal & Blend Weights Non-Convexity (Sum > 100%)
- **Location**: [`src/trading_bot/strategy/ensemble.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/strategy/ensemble.py) (Lines 111–130) and [`config/settings.yaml`](file:///c:/Users/moham/Desktop/makeway/config/settings.yaml)
- **Problem**: 
  - In `settings.yaml`, heuristic weights are defined: Momentum ($0.35$), Trend ($0.35$), Mean Reversion ($0.20$), News ($0.10$).
  - In `ensemble.py`, `base_rule` calculates:
    $$\text{base\_rule} = 0.35 \cdot \text{mom} + 0.35 \cdot \text{trend} + 0.20 \cdot \text{mr} + 0.10 \cdot \text{news}$$
    Then it blends `DipBuyingStrategy`:
    $$\text{rule\_score} = 0.70 \cdot \text{base\_rule} + 0.30 \cdot \text{dip\_score}$$
    And then blends Machine Learning:
    $$\text{ensemble\_score} = (1 - \text{ml\_weight}) \cdot \text{rule\_score} + \text{ml\_weight} \cdot \text{ml\_score}$$
  - Furthermore, in [`MomentumStrategy`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/strategy/signals.py), the raw momentum score is multiplied by `vol_mult` (up to $1.3\times$). In `DipBuyingStrategy`, the score can exceed $1.0$.
  - When combined without normalization, effective weights across components become arbitrary and non-convex (summing to $120\%+$).
- **Impact**: Distorts conviction calibration, skews position sizing, and produces unstable threshold transitions.
- **Remedy (Phase 4)**: Eliminate arbitrary hand-tuned weights. Implement meta-labeling stacking using an out-of-fold regularized Logistic Regression/Softmax classifier that empirically learns the optimal convex combination.

---

### Finding 2: Fixed-Percentage Exits (+15% / -5%) Cause Volatility Mismatch
- **Location**: [`src/trading_bot/risk/exposure.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/risk/exposure.py)
- **Problem**:
  - Exits are hardcoded to `take_profit_pct = 0.15` (+15%) and `per_position_stop_loss_pct = 0.05` (-5%).
  - Financial asset price volatility is highly heterogeneous across stocks and dynamic over time.
  - For a high-beta stock like `PLNX` (daily volatility $> 4\%$), a $-5\%$ stop loss is triggered by intraday noise within 1–2 bars, resulting in premature stop-outs on winning trends.
  - Conversely, for a low-beta stock like `AERX` (daily volatility $< 1\%$), a $15\%$ take profit takes months to materialize, while a $-5\%$ stop represents a catastrophic 5-standard-deviation tail event.
- **Impact**: Suboptimal Sharpe ratio, high turnover from noise-triggered stops, and failure to capture fat-tailed trend runs.
- **Remedy (Phase 5)**: Replace fixed percentages with dynamic **ATR-based volatility multiples**:
  - Stop Loss: $\text{Entry} - k_{\text{SL}} \times \text{ATR}_{14}$
  - Take Profit: $\text{Entry} + k_{\text{TP}} \times \text{ATR}_{14}$
  - Trailing Stop: $\text{Peak} - k_{\text{trail}} \times \text{ATR}_{14}$
  - Time Stop: Exit if neither barrier is touched within $N_{\text{max}}$ bars.

---

### Finding 3: Naive Fixed-Horizon Forward Return Labeling ($R_{t+3} > 20\text{ bps}$)
- **Location**: [`src/trading_bot/ml/dataset.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/ml/dataset.py) (Lines 98–106)
- **Problem**:
  - The current target definition evaluates only endpoint price:
    $$y_t = \mathbb{I}\left( \frac{C_{t+3} - C_t}{C_t} > 0.0020 \right)$$
  - **Path Ignorance**: If the price plunges $-15\%$ on day $t+1$ (triggering a stop-out or margin call) and subsequently rebounds to $+0.30\%$ on day $t+3$, the fixed-horizon formula labels this as a positive class ($y_t = 1$). In actual trading, the strategy incurred a $-15\%$ loss on day $t+1$.
  - **Fixed Hurdle**: $20\text{ bps}$ is constant regardless of market regime, volatility, or spread.
  - **Single Direction**: Binary classification $\{0, 1\}$ fails to label short-selling opportunities ($\{-1, 0, 1\}$).
- **Impact**: The model learns to buy patterns that suffer severe path drawdowns, degrading real-world Sharpe ratio.
- **Remedy (Phase 3)**: Implement **Marcos Lopez de Prado's Triple-Barrier Method**:
  - Upper barrier: $+k_1 \times \sigma_t$ (profit taking)
  - Lower barrier: $-k_2 \times \sigma_t$ (stop loss)
  - Vertical barrier: $N$ bars (expiration)
  - Label $y_t \in \{-1, 0, 1\}$ according to which barrier was touched first.
  - Add sample uniqueness and return magnitude weighting to eliminate sample redundancy.

---

### Finding 4: Lack of Purging and Embargo in Cross-Validation
- **Location**: [`src/trading_bot/ml/dataset.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/ml/dataset.py) and [`src/trading_bot/ml/pipeline.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/ml/pipeline.py)
- **Problem**:
  - Walk-forward splits currently partition data without purging overlapping label spans.
  - Since labels span $H$ bars into the future, samples within $H$ bars of the train/test boundary share common forward return data.
  - Autoregressive memory in financial returns causes informational leakage across consecutive folds.
- **Impact**: Artificially inflated out-of-sample metrics; models fail when deployed live.
- **Remedy (Phase 1 & 4)**: Implement **Purged & Embargoed Cross-Validation**:
  - Purge: Discard training samples whose event horizons overlap with test fold boundaries.
  - Embargo: Apply a $1\%$–$2\%$ sample embargo immediately following test sets to prevent autoregressive leakage.

---

### Finding 5: Backtester Runway Compounding & Slippage Impact Omission
- **Location**: [`src/trading_bot/research/backtest.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/research/backtest.py)
- **Problem**:
  - Slippage is modeled as a static flat 5 bps, independent of trade size or market volume. In reality, market impact scales non-linearly with order size relative to average daily volume (Square Root Law: $I \propto \sigma \sqrt{Q / V}$).
  - In short selling, cash was previously credited without accounting for margin maintenance liabilities, creating compounding distortion over hundreds of bars.
  - Taxes (Securities Transaction Tax - STT, exchange turnover fees, GST) and execution latency were omitted.
- **Impact**: Distorts strategy comparison and overestimates high-turnover strategy returns.
- **Remedy (Phase 1)**: Build a dedicated, event-driven backtesting engine that enforces:
  - Realistic non-linear slippage: $\text{Slippage} = \text{HalfSpread} + \gamma \cdot \left(\frac{Q}{V}\right)^{0.5}$
  - Mandatory transaction frictions: Brokerage (₹20/order), DP charge (₹15.34/sell), STT (0.025% on intraday/short, 0.1% on delivery), and GST.
  - Configurable execution latency (simulating 50–500 ms fill delay).
  - Identical code path as live order management.

---

### Finding 6: Missing Cross-Sectional Alpha & Breadth Features
- **Location**: [`src/trading_bot/features/indicators.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/features/indicators.py)
- **Problem**:
  - The feature space consists exclusively of single-stock time-series indicators.
  - It completely misses cross-sectional relative strength (which platform stocks are outperforming their peers today?).
  - It misses market breadth (% of stocks above 20 EMA, advance-decline ratio across the 10 assets).
  - It misses distributional shape features (rolling skewness, rolling kurtosis, Parkinson realized volatility, distance from 52-week/20-day high-low).
- **Impact**: The model cannot distinguish between idiosyncratic company alpha and broader market tide movements.
- **Remedy (Phase 2)**: Expand the feature pipeline to compute cross-sectional ranks, market breadth, realized volatility, and distributional higher moments point-in-time.

---

### Finding 7: Lack of Operational Safety Controls
- **Location**: [`src/trading_bot/risk/limits.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/risk/limits.py) & [`main.py`](file:///c:/Users/moham/Desktop/makeway/src/trading_bot/main.py)
- **Problem**:
  - Missing daily loss limit (circuit breaker for the single day session).
  - Missing tick timestamp staleness check (protecting against execution on stale network buffered ticks).
  - Missing feature/prediction drift monitoring (detecting when live market distributions deviate from training data).
  - Missing idempotent client order IDs to prevent duplicate order submissions during network timeouts.
- **Impact**: Risk of executing on outdated quotes or stacking losses on unpredictable market disconnects.
- **Remedy (Phase 6)**: Implement strict daily loss limit, stale-data guard ($>30$s skip), client idempotency tokens, and Population Stability Index (PSI) drift monitoring.

---

## 3. Architecture Upgrade Roadmap

```
Phase 0: System Audit (AUDIT.md) ───────────► [COMPLETE]
               │
Phase 1: Event-Driven Backtest Engine ───────► Realistic costs, slippage impact, purged CV
               │
Phase 2: Expanded Feature Suite ─────────────► Cross-sectional ranks, breadth, RV, skew/kurt
               │
Phase 3: Triple-Barrier Labeling ────────────► Volatility bounds, path dependency, sample weights
               │
Phase 4: Multi-Model Machine Learning ───────► LightGBM, XGBoost, CatBoost, HMM regimes, Stacking
               │
Phase 5: Dynamic Sizing & ATR Exits ─────────► Fractional Kelly, ATR stops/trailing/time exits
               │
Phase 6: Safety, Drift & Pre-Trade Risk ─────► Daily loss limit, stale tick guard, PSI drift
               │
Phase 7: Live Paper Trading & Final Report ──► 10-stock simulation, honest risk disclosure
```

---

## 4. Phase 0 Audit Conclusion & Sign-Off

The system audit is complete. All identified bugs and design flaws have been cataloged with clear mathematical remedies. We proceed immediately to **Phase 1: Backtest Engine**, building the event-driven validation harness first to ensure every subsequent model, feature, and exit is rigorously verified under realistic market frictions.
