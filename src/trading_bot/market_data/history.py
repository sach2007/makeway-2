"""Historical data store and rolling window manager for symbols."""
from __future__ import annotations
import logging
from collections import deque
from typing import Optional, TYPE_CHECKING
import pandas as pd

from trading_bot.api.schemas import HistoryBar, MarketTick

if TYPE_CHECKING:
    from trading_bot.api.client import TradingAPIClient

logger = logging.getLogger(__name__)


class SymbolHistoryStore:
    """Maintains rolling historical bars for each symbol to power indicator calculations."""

    def __init__(self, max_bars: int = 300):
        self.max_bars = max_bars
        # Map: ticker -> deque of dicts with keys: date, open, high, low, close, volume
        self._bars: dict[str, deque[dict]] = {}
        # Cached dataframes for fast reading
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._cache_dirty: dict[str, bool] = {}

    def warm_up_symbol(self, ticker: str, history_records: list[dict]) -> int:
        """Initialize symbol history from REST API response."""
        ticker = ticker.upper()
        if ticker not in self._bars:
            self._bars[ticker] = deque(maxlen=self.max_bars)
        
        count = 0
        for rec in history_records:
            try:
                bar = {
                    "date": str(rec.get("date", "")),
                    "open": float(rec.get("open", 0.0)),
                    "high": float(rec.get("high", 0.0)),
                    "low": float(rec.get("low", 0.0)),
                    "close": float(rec.get("close", 0.0)),
                    "volume": float(rec.get("volume", 0.0)),
                }
                if bar["close"] > 0:
                    self._bars[ticker].append(bar)
                    count += 1
            except (ValueError, TypeError):
                continue
                
        self._cache_dirty[ticker] = True
        logger.info(f"Warmed up {count} historical bars for {ticker}")
        return count

    def warm_up_universe(self, client: TradingAPIClient, tickers: list[str], days: int = 365) -> dict[str, int]:
        """Warm up all symbols in the active tradable universe."""
        results = {}
        for ticker in tickers:
            try:
                records = client.get_history(ticker, days=days)
                count = self.warm_up_symbol(ticker, records)
                results[ticker] = count
            except Exception as e:
                logger.warning(f"Failed to warm up history for {ticker}: {e}")
                results[ticker] = 0
        return results

    def on_tick(self, tick: MarketTick) -> None:
        """Process an incoming tick. If tick date matches last bar date, update it; else append new bar."""
        ticker = tick.ticker.upper()
        if ticker not in self._bars:
            self._bars[ticker] = deque(maxlen=self.max_bars)

        bar_dict = {
            "date": str(tick.simulated_date or tick.timestamp or ""),
            "open": tick.open,
            "high": tick.high,
            "low": tick.low,
            "close": tick.close,
            "volume": tick.volume,
        }

        # Check if we should update existing bar (same date) or append a new daily bar
        if self._bars[ticker] and self._bars[ticker][-1]["date"] == bar_dict["date"]:
            last = self._bars[ticker][-1]
            last["high"] = max(last["high"], bar_dict["high"])
            last["low"] = min(last["low"], bar_dict["low"])
            last["close"] = bar_dict["close"]
            last["volume"] += bar_dict["volume"]
        else:
            self._bars[ticker].append(bar_dict)

        self._cache_dirty[ticker] = True

    def get_dataframe(self, ticker: str) -> pd.DataFrame:
        """Return historical DataFrame for ticker with columns: [date, open, high, low, close, volume]."""
        ticker = ticker.upper()
        if ticker not in self._bars or not self._bars[ticker]:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        if self._cache_dirty.get(ticker, True) or ticker not in self._df_cache:
            df = pd.DataFrame(list(self._bars[ticker]))
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["open"] = pd.to_numeric(df["open"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            self._df_cache[ticker] = df
            self._cache_dirty[ticker] = False

        return self._df_cache[ticker]

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Return the most recent close price for a ticker."""
        ticker = ticker.upper()
        if ticker in self._bars and self._bars[ticker]:
            return self._bars[ticker][-1]["close"]
        return None

    def get_bar_count(self, ticker: str) -> int:
        """Return number of cached bars for ticker."""
        ticker = ticker.upper()
        return len(self._bars.get(ticker, []))
