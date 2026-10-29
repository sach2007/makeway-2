"""End-to-end ML research pipeline: training, walk-forward validation, and artifact persistence."""
from __future__ import annotations
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd

from trading_bot.ml.dataset import FEATURE_COLUMNS, TimeSeriesFeaturePipeline
from trading_bot.ml.models import MLModelWrapper, get_candidate_models
from trading_bot.portfolio.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


class MLPipelineRunner:
    """Manages the full machine learning training, walk-forward evaluation, and model artifact saving."""

    def __init__(
        self,
        output_dir: str = "models",
        target_horizon: int = 3,
        cost_hurdle_bps: float = 20.0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_pipeline = TimeSeriesFeaturePipeline(
            target_horizon=target_horizon,
            cost_hurdle_bps=cost_hurdle_bps
        )

    def simulate_model_trading(
        self,
        model: MLModelWrapper,
        test_df: pd.DataFrame,
        buy_threshold: float = 0.53,
        fee_bps: float = 5.0,
        slippage_bps: float = 5.0
    ) -> dict[str, float]:
        """Conservative per-ticker trade proxy, not a substitute for full OMS replay.

        Each ticker has its own holding/cooldown state. A position is held for the
        target horizon, and no overlapping trades are opened in the same ticker.
        Flat contest brokerage and DP charges are included; slippage remains a
        configurable sensitivity assumption. Metrics are labelled as a research
        proxy in reports rather than a live-performance forecast.
        """
        empty = {"final_wealth": 1_000_000.0, "net_return_pct": 0.0, "sharpe_ratio": 0.0,
                 "max_drawdown_pct": 0.0, "total_trades": 0, "win_rate_pct": 0.0, "profit_factor": 0.0}
        if test_df.empty or "ticker" not in test_df.columns:
            return empty
        work = test_df.sort_values(["date", "ticker"]).copy()
        probs = model.predict_proba(work[FEATURE_COLUMNS].values)
        work["_prob"] = probs
        per_ticker_returns = []
        realized = []
        # Brokerage is charged as a flat rupee amount below; only the explicit
        # slippage argument is treated as a proportional execution assumption.
        slippage_rate = max(0.0, slippage_bps / 10000.0)
        initial = 1_000_000.0
        ticker_returns = []
        for ticker, group in work.groupby("ticker", sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            wealth = initial
            blocked_until = -1
            returns = []
            for i, row in group.iterrows():
                if i <= blocked_until or float(row["_prob"]) < buy_threshold:
                    returns.append(0.0)
                    continue
                fwd_ret = float(row["forward_return"])
                if not np.isfinite(fwd_ret):
                    returns.append(0.0)
                    continue
                notional = wealth * 0.20
                # Entry and exit are two executed orders; DP applies on delivery sale.
                friction = (2.0 * 20.0) + 15.34 + (notional * 2.0 * slippage_rate)
                pnl = notional * fwd_ret - friction
                wealth += pnl
                realized.append(pnl)
                returns.append(pnl / max(wealth - pnl, 1.0))
                blocked_until = i + max(1, int(self.feature_pipeline.target_horizon)) - 1
            ticker_returns.append((wealth - initial) / initial)
            per_ticker_returns.extend(returns)
        if not ticker_returns:
            return empty
        # Equal-weighted average of independent ticker simulations, avoiding the
        # previous shared `holding` flag that mixed symbols into one fake position.
        total_return = float(np.mean(ticker_returns))
        final_wealth = initial * (1.0 + total_return)
        equity = [initial]
        for r in per_ticker_returns:
            equity.append(equity[-1] * (1.0 + r / max(1, len(ticker_returns))))
        _, dd_pct = PerformanceMetrics.calculate_max_drawdown(equity)
        sharpe = PerformanceMetrics.calculate_sharpe(per_ticker_returns)
        trade_metrics = PerformanceMetrics.calculate_trade_metrics(realized)
        return {
            "final_wealth": round(final_wealth, 2),
            "net_return_pct": round(total_return * 100.0, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(dd_pct) * 100.0, 2),
            "total_trades": len(realized),
            "win_rate_pct": trade_metrics["win_rate_pct"],
            "profit_factor": trade_metrics["profit_factor"],
        }

    def walk_forward_cv(
        self,
        dataset: pd.DataFrame,
        n_splits: int = 3
    ) -> dict[str, list[float]]:
        """Expanding-window, date-grouped CV with purged forward-label boundaries."""
        results: dict[str, list[float]] = {}
        splits = list(TimeSeriesFeaturePipeline.walk_forward_splits(
            dataset, n_splits=n_splits, min_train_ratio=0.40,
            purge_days=self.feature_pipeline.target_horizon
        ))
        for split_idx, (train_slice, test_slice) in enumerate(splits):
            if len(test_slice) < 20 or train_slice["target"].nunique() < 2:
                continue
            X_tr = train_slice[FEATURE_COLUMNS].values
            y_tr = train_slice["target"].values
            X_te = test_slice[FEATURE_COLUMNS].values
            y_te = test_slice["target"].values
            candidates = get_candidate_models()
            for key, model in candidates.items():
                results.setdefault(key, [])
                try:
                    model.fit(X_tr, y_tr, feature_names=FEATURE_COLUMNS)
                    eval_res = model.evaluate(X_te, y_te)
                    results[key].append(eval_res["roc_auc"])
                except Exception as e:
                    logger.warning("Fold %d failed for %s: %s", split_idx, key, e)
        return results

    def train_and_evaluate(
        self,
        symbol_dfs: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Execute end-to-end ML training, evaluation, walk-forward validation, and artifact generation."""
        logger.info(f"Building time-series dataset from {len(symbol_dfs)} symbol dataframes...")
        dataset = self.feature_pipeline.build_multi_symbol_dataset(symbol_dfs)

        if dataset.empty or len(dataset) < 80:
            logger.warning("Insufficient data to train ML models (< 80 usable rows).")
            return {"error": "INSUFFICIENT_DATA", "status": "failed"}

        # 1. Chronological Split (No Leakage)
        train_df, val_df, test_df = TimeSeriesFeaturePipeline.chronological_split(dataset, 0.65, 0.15)

        if train_df.empty or val_df.empty or test_df.empty or train_df["target"].nunique() < 2:
            logger.error("Date-purged split is too small or has only one training class; refusing to overwrite champion model.")
            return {"error": "INSUFFICIENT_PURGED_SPLIT", "status": "failed"}

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df["target"].values

        X_val = val_df[FEATURE_COLUMNS].values
        y_val = val_df["target"].values

        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df["target"].values

        # 2. Walk-Forward Cross Validation Scores
        logger.info("Computing Walk-Forward Cross Validation over expanding windows...")
        wf_scores = self.walk_forward_cv(dataset, n_splits=3)

        # 3. Train and Evaluate Candidate Models on Held-Out Test Set
        candidates = get_candidate_models()
        evaluation_results = []
        best_model: Optional[MLModelWrapper] = None
        best_model_key: Optional[str] = None
        best_score = -999.0

        for key, model in candidates.items():
            logger.info(f"Fitting model: {model.model_name} (GPU: {model.uses_gpu}) on {len(X_train)} train rows...")
            try:
                model.fit(X_train, y_train, feature_names=FEATURE_COLUMNS)
                # Model selection uses validation only. The test set is untouched
                # until a champion has been selected.
                pred_metrics = model.evaluate(X_val, y_val)
            except Exception as exc:
                logger.warning("Skipping candidate %s due to training/evaluation error: %s", key, exc)
                continue

            trade_metrics = self.simulate_model_trading(model, val_df)
            wf_mean_auc = round(float(np.mean(wf_scores.get(key, [pred_metrics["roc_auc"]]))), 4)
            combined_eval = {
                "model_key": key,
                "selection_split": "validation",
                "walk_forward_mean_auc": wf_mean_auc,
                **pred_metrics,
                **trade_metrics
            }
            evaluation_results.append(combined_eval)

            composite_score = trade_metrics["sharpe_ratio"] + (wf_mean_auc * 2.0)
            if composite_score > best_score:
                best_score = composite_score
                best_model = model
                best_model_key = key

        if best_model is None:
            return {"error": "NO_VALID_MODEL", "status": "failed"}

        # Refit the validation-selected champion on train + validation, then
        # evaluate exactly once on the untouched final test window.
        train_val_df = pd.concat([train_df, val_df], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
        best_model.fit(train_val_df[FEATURE_COLUMNS].values, train_val_df["target"].values, feature_names=FEATURE_COLUMNS)
        champion_test_metrics = best_model.evaluate(X_test, y_test)
        champion_test_trading = self.simulate_model_trading(best_model, test_df)

        # 4. Baseline Strategy Comparison on Test Split (Buy & Hold)
        test_returns = test_df["close"].pct_change().dropna().values
        baseline_ret = (test_df["close"].iloc[-1] - test_df["close"].iloc[0]) / test_df["close"].iloc[0] * 100.0
        baseline_sharpe = PerformanceMetrics.calculate_sharpe(list(test_returns))
        _, baseline_dd_pct = PerformanceMetrics.calculate_max_drawdown(list(test_df["close"].values))

        baseline_result = {
            "model_key": "baseline_buy_and_hold",
            "model": "Baseline (Buy & Hold)",
            "uses_gpu": False,
            "walk_forward_mean_auc": 0.50,
            "accuracy": 0.50,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.50,
            "brier_score": 0.25,
            "final_wealth": round(1_000_000.0 * (1.0 + baseline_ret / 100.0), 2),
            "net_return_pct": round(baseline_ret, 2),
            "sharpe_ratio": round(baseline_sharpe, 2),
            "max_drawdown_pct": round(baseline_dd_pct * 100.0, 2),
            "total_trades": 1,
            "win_rate_pct": 100.0 if baseline_ret > 0 else 0.0,
            "profit_factor": 1.0,
        }
        evaluation_results.append(baseline_result)

        # 5. Save Artifacts if Champion Model selected
        champion_name = best_model.model_name if best_model else "None"
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_note": "ML trade metrics are a per-ticker research proxy with flat brokerage and DP charge; use the event-driven backtester for portfolio-level validation. Not a live return forecast.",
            "target_horizon_days": self.feature_pipeline.target_horizon,
            "cost_hurdle_bps": self.feature_pipeline.cost_hurdle * 10000.0,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "champion_model": champion_name,
            "champion_model_key": best_model_key,
            "champion_selection_split": "validation",
            "champion_test_metrics": {**champion_test_metrics, **champion_test_trading},
            "champion_uses_gpu": best_model.uses_gpu if best_model else False,
            "models_comparison": evaluation_results
        }

        # Save model artifacts
        if best_model:
            model_path = self.output_dir / "champion_model.joblib"
            joblib.dump(best_model, model_path)

            features_path = self.output_dir / "feature_names.json"
            with open(features_path, "w", encoding="utf-8") as f:
                json.dump(FEATURE_COLUMNS, f, indent=2)

            meta_path = self.output_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            report_path = self.output_dir / "evaluation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=2)

            logger.info(f"Champion model [{champion_name}] saved to {model_path}")

        return report
