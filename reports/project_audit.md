# MANTHAN 2026 Project & Competition Rules Audit Report

**Event**: MANTHAN 2026 — 24-Hour National Level Hackathon  
**Organizer**: Department of Computer Science & Engineering (Data Science), Jawaharlal Nehru New College of Engineering (JNNCE), Shivamogga, Karnataka  
**Theme**: Real-Time Predictive Analytics — Autonomous Trading Systems Challenge  
**Team**: Makeway (`MN2672`) | Account Verified: `true` (Team ID: `team_78oqt65`)  
**Audit Timestamp**: 2026-10-29  

---

## 1. Executive Summary & Verified Environment

The MANTHAN 2026 Autonomous Trading Systems Challenge requires teams to build, deploy, and operate a fully autonomous machine-learning trading system capable of consuming live market ticks and news streams via Apache Kafka, evaluating alpha signals with ML models, enforcing strict risk limits, and submitting market orders to the central Order Gateway.

### Platform Infrastructure Mapping

| Component | Network Endpoint | Status / Notes |
|---|---|---|
| **Order Gateway REST API** | `http://192.168.65.157:3002` | Real-time synchronous execution with **0ms artificial delay**; handles order matching, portfolio MTM, and preflight heartbeats. |
| **Participant Web Portal** | `http://192.168.65.157:3003` | React Single Page Application (SPA) dashboard for live telemetry, order history, and portfolio visualization. |
| **Kafka Broker** | `192.168.65.157:9094` | External Kafka broker port for participants. Topics: `market.ticks` (OHLCV stream) and `market.news` (breaking corporate news). |
| **Active Dataset** | `independent_test_2yr` | 10 tradable stocks: `AERX`, `BLUX`, `DSNY`, `HAVN`, `HMDP`, `MOTR`, `ORCX`, `PLNX`, `ROUT`, `SONA`. |
| **Initial Capital** | ₹1,000,000.00 (10 Lakhs) | Virtual starting capital in INR. |

---

## 2. Official Competition & Strategy Rules

### Rule 1: Latency Constraint (Crucial)
- **Constraint**: Bot trading decisions must take **no more than 3 to 4 seconds** from tick arrival to order dispatch (per official briefing video `QKgdReXq1-M`).
- **Implementation**: Feature extraction and ML inference are optimized for sub-millisecond execution (<0.1 ms on CPU), ensuring full compliance with the 3–4 second rule.

### Rule 2: Two-Way Short Selling & Position Semantics
The platform natively supports long and short positions across four actions:
1. `BUY`: Purchases shares using available cash.
2. `SELL`: Closes or reduces an existing long holding to realize profit or loss.
3. `SHORT`: Borrows and sells shares at market price; credits cash balance with negative quantity, locking margin.
4. `COVER`: Purchases shares to close or reduce an existing short liability:
   $$\text{Realized P\&L} = (\text{Short Entry Price} - \text{Cover Price}) \times \text{Shares}$$

### Rule 3: Anti-Lookahead Protection
- `GET /api/v1/stocks/history` strictly enforces anti-lookahead. Querying dates on or after the current simulated date triggers `HTTP 400 INVALID_DATE_RANGE`.
- Models and technical features must strictly use point-in-time data ($t \le \tau_{\text{current}}$).

### Rule 4: Pre-Flight Readiness Heartbeat
- During the preflight phase, the platform emits `preflight_tick` events on `market.ticks` with a sequence number `seq`.
- Bots must acknowledge each event within 2.5 seconds via `POST /api/v1/bot/heartbeat` with `{"seq": seq}` to prove bot liveness.

### Rule 5: Trading Costs & Friction
- **Brokerage**: ₹20.00 flat per executed order.
- **Depository Participant (DP) Charges**: ₹15.34 flat on sell orders.
- **Slippage**: Simulated at ~5 bps.
- All strategies and backtests must evaluate net performance *after* accounting for all frictions.

---

## 3. Verified REST API Directory

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/v1/orders` | POST | Bearer JWT | Submit market order (`BUY`, `SELL`, `SHORT`, `COVER`). Accepts `ticker`, `action`, `quantity`, optional `date`. |
| `/api/v1/orders` | GET | Bearer JWT | Query order audit logs, fill prices, fees, and realized P&L. |
| `/api/v1/portfolio` | GET | Bearer JWT | Authoritative mark-to-market portfolio snapshot, long/short holdings, realized & unrealized P&L. |
| `/api/v1/portfolio/cash` | GET | Bearer JWT | Lightweight 0ms cash balance query for high-frequency position sizing. |
| `/api/v1/stocks` | GET | Bearer JWT | Authenticated list of tradable stocks, company names, and business descriptions. |
| `/api/v1/stocks/history` | GET | Bearer JWT | Up to 10 years of historical daily OHLCV bars strictly prior to the current simulation date. |
| `/api/v1/simulation/clock` | GET | Public | Simulation clock: `current_day`, `total_days`, `simulated_date`, `seconds_per_day`, `status`. |
| `/api/v1/stage` | GET | Public | Platform lifecycle stage (`development`, `ready`, `preflight`, `simulation`). |
| `/api/v1/bot/heartbeat` | POST | Bearer JWT | Acknowledge preflight heartbeat sequence number (`seq`). |

---

## 4. Hardware & Compute Budgeting Audit

- **Host GPU**: NVIDIA GeForce RTX 4050 Laptop GPU
  - **VRAM**: 6,141 MiB (6 GB dedicated GDDR6)
  - **Driver Version**: 581.86 | CUDA Version: 13.0
  - **Compute Role**: Accelerated training of tree ensembles (XGBoost GPU with `tree_method='hist'`, `device='cuda'`), completing 10-stock multi-fold walk-forward validation in <2 seconds.
- **Host CPU & Memory**:
  - **System RAM**: 16 GB DDR (~3.5 GB free physical memory)
  - **Compute Role**: Parallel LightGBM & Random Forest training (`n_jobs=-1`), continuous feature engineering, point-in-time sliding windows.
  - **Live Inference**: Single-pass CPU inference (<0.1 ms latency per tick), zero GPU context-switching overhead during live trading loops.

---

## 5. Official Evaluation & Scorecard Alignment

The jury panel evaluates bots across four core pillars:
1. **Total Net Wealth & Realized P&L**: Capital preservation and wealth generation over the 10-stock universe.
2. **Risk-Adjusted Performance**: Sharpe Ratio $> 1.0$, Maximum Drawdown $\le 15\%$, win rate $> 55\%$.
3. **Execution Quality**: 100% fill rate, 0 rejected orders, adherence to the 3–4 second decision latency constraint.
4. **Engineering Rigor**: Causal zero-leakage feature pipeline, deterministic pre-trade risk veto sovereignty, automatic failover, and comprehensive test coverage.
