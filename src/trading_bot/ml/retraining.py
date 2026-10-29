"""Continuous online model retraining module for real-time model adaptation without trading downtime."""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd

from trading_bot.ml.dataset import FEATURE_COLUMNS, TimeSeriesFeaturePipeline
from trading_bot.ml.models import MLModelWrapper, get_candidate_models
from trading_bot.ml.pipeline import MLPipelineRunner

logger = logging.getLogger(__name__)


class ContinuousRetrainer:
    """Manages background online incremental training when new market bars accumulate."""

    def __init__(
        self,
        models_dir: str = "models",
        bars_trigger_retrain: int = 50,
        target_horizon: int = 3,
        cost_hurdle_bps: float = 20.0
    ):
        self.models_dir = Path(models_dir)
        self.bars_trigger_retrain = bars_trigger_retrain
        self.target_horizon = target_horizon
        self.cost_hurdle_bps = cost_hurdle_bps
        self._accumulated_bars: dict[str, list[dict]] = {}
        self._is_retraining = False
        self._lock = threading.Lock()
        self.retrain_count = 0
        self.last_retrain_time = 0.0

    def on_bar_closed(self, ticker: str, bar_data: dict) -> None:
        """Register a newly closed bar from the market stream."""
        with self._lock:
            if ticker not in self._accumulated_bars:
                self._accumulated_bars[ticker] = []
            self._accumulated_bars[ticker].append(bar_data)

            # Check if total new bars across tickers exceed trigger threshold
            total_new_bars = sum(len(bars) for bars in self._accumulated_bars.values())
            if total_new_bars >= self.bars_trigger_retrain and not self._is_retraining:
                logger.info(f"Triggering continuous online model retraining ({total_new_bars} new bars accumulated)...")
                self._is_retraining = True
                # Run retraining in background thread to avoid blocking live tick path (<3-4s latency rule)
                thread = threading.Thread(target=self._run_background_retrain, daemon=True)
                thread.start()

    def _run_background_retrain(self) -> None:
        """Execute background retraining and atomically update model artifacts."""
        try:
            t0 = time.perf_counter()
            data_dir = Path("data/processed")
            csv_files = list(data_dir.glob("*.csv"))

            symbol_dfs = {}
            for f in csv_files:
                df = pd.read_csv(f)
                ticker = f.stem
                # Append recently accumulated bars
                with self._lock:
                    new_bars = self._accumulated_bars.get(ticker, [])
                    if new_bars:
                        new_df = pd.DataFrame(new_bars)
                        df = pd.concat([df, new_df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
                symbol_dfs[ticker] = df

            runner = MLPipelineRunner(
                output_dir=str(self.models_dir),
                target_horizon=self.target_horizon,
                cost_hurdle_bps=self.cost_hurdle_bps
            )
            report = runner.train_and_evaluate(symbol_dfs)

            elapsed = time.perf_counter() - t0
            self.retrain_count += 1
            self.last_retrain_time = time.time()
            logger.info(
                f"Continuous retraining #{self.retrain_count} completed in {elapsed:.2f}s! "
                f"Champion: {report.get('champion_model')}"
            )

            # Reset accumulated counter after successful retrain
            with self._lock:
                self._accumulated_bars.clear()

        except Exception as e:
            logger.error(f"Background continuous retraining encountered error: {e}")
        finally:
            self._is_retraining = False
