"""Script to run historical backtest comparisons and generate trade ledger CSV across all platform stocks."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from trading_bot.config import Settings
from trading_bot.research.backtest import BacktestEngine


def main():
    print("=" * 85)
    print("      MANTHAN 2026 STRATEGY RESEARCH & EVENT-DRIVEN BACKTEST ENGINE      ")
    print("=" * 85)

    data_dir = Path("data/processed")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("ERROR: No processed data found in data/processed/. Run scripts/ingest_data.py first.")
        return

    print(f"Loading {len(csv_files)} processed stock datasets for event-driven backtesting...")

    engine = BacktestEngine(
        initial_capital=1_000_000.0,
        fee_bps=5.0,
        slippage_bps=5.0,
        dp_charge=15.34,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )

    strategies = ["baseline", "momentum", "trend", "mean_reversion", "ensemble"]
    results = []

    for f in csv_files:
        ticker = f.stem
        df = pd.read_csv(f)
        df["close"] = pd.to_numeric(df["close"])
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["volume"] = pd.to_numeric(df["volume"])

        print(f"Backtesting {ticker} ({len(df)} bars)...")

        for strat in strategies:
            res = engine.run_single_symbol(ticker, df, strategy_mode=strat)
            res["ticker"] = ticker
            results.append(res)

    if not results:
        print("No backtest results to display.")
        return

    # 1. Display Benchmark Results Table
    res_df = pd.DataFrame(results)
    cols = [
        "ticker", "strategy", "final_wealth", "total_return_pct",
        "sharpe_ratio", "max_drawdown_pct", "total_trades", "win_rate_pct"
    ]
    print("\n" + "=" * 95)
    print("                         BENCHMARK RESULTS TABLE")
    print("=" * 95)
    print(res_df[cols].to_string(index=False))
    print("=" * 95)

    # 2. Aggregate by strategy
    agg = res_df.groupby("strategy").agg({
        "total_return_pct": "mean",
        "sharpe_ratio": "mean",
        "max_drawdown_pct": "mean",
        "win_rate_pct": "mean",
        "total_trades": "sum"
    }).reset_index()

    print("\n" + "=" * 80)
    print("                  AVERAGE PERFORMANCE ACROSS ALL ASSETS")
    print("=" * 80)
    print(agg.to_string(index=False))
    print("=" * 80)

    # 3. Export trade ledger to reports/trade_ledger.csv
    ledger_df = pd.DataFrame(engine.all_trade_records)
    ledger_path = reports_dir / "trade_ledger.csv"
    ledger_df.to_csv(ledger_path, index=False)
    print(f"\n[OK] Saved trade-by-trade ledger ({len(ledger_df)} executed trades) to {ledger_path}")

    # 4. Export backtest report to reports/backtest_report.md
    report_path = reports_dir / "backtest_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MANTHAN 2026 Event-Driven Strategy Backtest Report\n\n")
        f.write(f"- **Execution Friction**: 5 bps brokerage fee + 5 bps slippage per side (10 bps round trip)\n")
        f.write(f"- **Depository Participant Charges**: ₹15.34 flat per sell execution\n")
        f.write(f"- **Risk Rules Enforced**: -5% Stop-Loss, +15% Take-Profit Target\n")
        f.write(f"- **Asset Universe**: 10 platform stocks ({len(csv_files)} datasets)\n")
        f.write(f"- **Total Executed Trades in Ledger**: {len(ledger_df):,}\n\n")
        f.write("### Strategy Average Performance Summary\n\n")
        try:
            f.write(agg.to_markdown(index=False))
        except Exception:
            f.write("```\n" + agg.to_string(index=False) + "\n```")
        f.write("\n\n### Per-Symbol Detailed Results\n\n")
        try:
            f.write(res_df[cols].to_markdown(index=False))
        except Exception:
            f.write("```\n" + res_df[cols].to_string(index=False) + "\n```")
        f.write("\n\n### Trade Ledger Sample (Latest 10 Trades)\n\n")
        if not ledger_df.empty:
            sample_cols = ["trade_id", "ticker", "strategy", "action", "entry_date", "exit_date", "shares", "entry_price", "exit_price", "net_pnl", "return_pct", "exit_reason"]
            avail_sample = [c for c in sample_cols if c in ledger_df.columns]
            try:
                f.write(ledger_df[avail_sample].tail(10).to_markdown(index=False))
            except Exception:
                f.write("```\n" + ledger_df[avail_sample].tail(10).to_string(index=False) + "\n```")
    print(f"[OK] Saved backtest report to {report_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
