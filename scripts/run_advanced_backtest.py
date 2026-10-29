"""Institutional multi-symbol backtest runner with purged walk-forward cross-validation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from trading_bot.research.backtester import EventDrivenBacktester
from trading_bot.research.purged_cv import PurgedWalkForwardCV


def main():
    print("=" * 95)
    print("      MANTHAN 2026 INSTITUTIONAL EVENT-DRIVEN BACKTEST & PURGED WALK-FORWARD CV      ")
    print("=" * 95)

    data_dir = Path("data/processed")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("ERROR: No data files found in data/processed/. Run scripts/ingest_data.py first.")
        return

    symbol_dfs = {}
    for f in csv_files:
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        symbol_dfs[f.stem] = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(symbol_dfs)} platform stocks ({sum(len(df) for df in symbol_dfs.values()):,} total bars).")

    # 1. Full Dataset Event-Driven Backtest with Realistic Frictions
    backtester = EventDrivenBacktester(
        initial_capital=1_000_000.0,
        brokerage_per_order=20.0,
        dp_charge_per_sell=15.34,
        stt_rate=0.00025,
        base_half_spread_bps=5.0,
        market_impact_gamma=0.10,
        latency_penalty_bps=2.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )

    strategies = ["baseline", "ensemble"]
    full_results = []

    print("\n--- Running Full Event-Driven Backtest (Baseline vs Regime Ensemble) ---")
    for strat in strategies:
        res = backtester.run(symbol_dfs, strategy_mode=strat)
        full_results.append(res)
        print(f"  [{strat.upper()}] Final Wealth: INR {res['final_wealth']:,.2f} | Net Return: {res.get('net_return_pct', res.get('total_return_pct', 0.0)):+.2f}% | "
              f"Sharpe: {res['sharpe_ratio']} | Sortino: {res['sortino_ratio']} | Calmar: {res['calmar_ratio']} | "
              f"Max DD: {res['max_drawdown_pct']}% | Total Frictions: INR {res.get('total_frictions_paid', 0.0):,.2f}")

    # 2. Purged & Embargoed Walk-Forward Cross-Validation (IS vs OOS)
    print("\n--- Executing 3-Fold Purged & Embargoed Walk-Forward Cross-Validation ---")
    cv = PurgedWalkForwardCV(n_splits=3, min_train_pct=0.45, label_horizon_bars=5, embargo_pct=0.02)

    # Use first symbol's date timeline as the canonical timeline
    sample_df = list(symbol_dfs.values())[0]
    wf_results = []

    for fold_idx, (train_timeline, test_timeline, meta) in enumerate(cv.split(sample_df)):
        train_start_d = train_timeline["date"].iloc[0]
        train_end_d = train_timeline["date"].iloc[-1]
        test_start_d = test_timeline["date"].iloc[0]
        test_end_d = test_timeline["date"].iloc[-1]

        # Slice symbol datasets for IS and OOS
        train_sym_dfs = {}
        test_sym_dfs = {}
        for sym, s_df in symbol_dfs.items():
            tr_slice = s_df[(s_df["date"] >= train_start_d) & (s_df["date"] <= train_end_d)]
            te_slice = s_df[(s_df["date"] >= test_start_d) & (s_df["date"] <= test_end_d)]
            if not tr_slice.empty:
                train_sym_dfs[sym] = tr_slice.reset_index(drop=True)
            if not te_slice.empty:
                test_sym_dfs[sym] = te_slice.reset_index(drop=True)

        # Run In-Sample
        is_bt = EventDrivenBacktester()
        is_res = is_bt.run(train_sym_dfs, strategy_mode="ensemble")

        # Run Out-of-Sample
        oos_bt = EventDrivenBacktester()
        oos_res = oos_bt.run(test_sym_dfs, strategy_mode="ensemble")

        wf_results.append({
            "fold": fold_idx + 1,
            "train_period": f"{str(train_start_d)[:10]} to {str(train_end_d)[:10]}",
            "test_period": f"{str(test_start_d)[:10]} to {str(test_end_d)[:10]}",
            "is_net_return_pct": is_res.get("total_return_pct", 0.0),
            "is_sharpe": is_res.get("sharpe_ratio", 0.0),
            "is_max_dd_pct": is_res.get("max_drawdown_pct", 0.0),
            "oos_net_return_pct": oos_res.get("total_return_pct", 0.0),
            "oos_sharpe": oos_res.get("sharpe_ratio", 0.0),
            "oos_max_dd_pct": oos_res.get("max_drawdown_pct", 0.0),
        })

        print(f"  Fold {fold_idx + 1} | Train: {meta['train_samples']} bars | Test: {meta['test_samples']} bars "
              f"(Purged: {meta['purged_bars']} bars, Embargo: {meta['embargo_bars']} bars)")
        print(f"    In-Sample:     Return: {is_res['total_return_pct']:+.2f}% | Sharpe: {is_res['sharpe_ratio']} | Max DD: {is_res['max_drawdown_pct']}%")
        print(f"    Out-of-Sample: Return: {oos_res['total_return_pct']:+.2f}% | Sharpe: {oos_res['sharpe_ratio']} | Max DD: {oos_res['max_drawdown_pct']}%")

    # 3. Save Report
    report_path = reports_dir / "institutional_backtest_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MANTHAN 2026 Institutional Event-Driven Backtest & Purged CV Report\n\n")
        f.write("## 1. Friction & Market Impact Specifications\n\n")
        f.write("- **Zero-Cost Impossibility**: Zero-cost results are hard-blocked by code assertions.\n")
        f.write("- **Brokerage Fee**: ₹20.00 flat per executed order.\n")
        f.write("- **Depository Participant (DP) Charge**: ₹15.34 flat per sell transaction.\n")
        f.write("- **Securities Transaction Tax (STT)**: 2.5 bps (0.025%) on all executed notionals.\n")
        f.write("- **Bid-Ask Spread**: 5 bps minimum half-spread.\n")
        f.write("- **Non-Linear Market Impact**: Square-root law $I = \\gamma \\sqrt{Q / V}$ with $\\gamma = 0.10$.\n")
        f.write("- **Execution Latency Penalty**: 2 bps slippage simulating order transmission delay.\n\n")
        
        f.write("## 2. Full Period Strategy Performance (Baseline vs Ensemble)\n\n")
        full_df = pd.DataFrame(full_results)[["strategy", "final_wealth", "total_return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor", "turnover_ratio", "total_frictions_paid"]]
        try:
            f.write(full_df.to_markdown(index=False))
        except Exception:
            f.write("```\n" + full_df.to_string(index=False) + "\n```")

        f.write("\n\n## 3. Purged & Embargoed Walk-Forward Validation (IS vs OOS Side-by-Side)\n\n")
        wf_df = pd.DataFrame(wf_results)
        try:
            f.write(wf_df.to_markdown(index=False))
        except Exception:
            f.write("```\n" + wf_df.to_string(index=False) + "\n```")

        f.write("\n\n## 4. Per-Regime Breakdown (Ensemble Strategy)\n\n")
        ens_regimes = full_results[1].get("per_regime_breakdown", {})
        if ens_regimes:
            reg_df = pd.DataFrame.from_dict(ens_regimes, orient="index").reset_index().rename(columns={"index": "regime"})
            try:
                f.write(reg_df.to_markdown(index=False))
            except Exception:
                f.write("```\n" + reg_df.to_string(index=False) + "\n```")

    print(f"\n[OK] Institutional backtest report saved to {report_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
