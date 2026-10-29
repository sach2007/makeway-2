"""Event normalizer converting raw Kafka/REST payloads into typed schemas."""
from __future__ import annotations
import logging
from typing import Any, Optional, Union

from trading_bot.api.schemas import MarketTick, NewsItem
from trading_bot.market_data.validator import DataValidator

logger = logging.getLogger(__name__)


class EventNormalizer:
    """Normalizes heterogeneous market events into canonical domain models."""
    
    @classmethod
    def normalize_tick_event(cls, raw: dict[str, Any]) -> Optional[MarketTick]:
        """Normalize a price_tick event.
        Handles both nested 'data' dictionaries and flat structures.
        """
        try:
            timestamp = raw.get("timestamp")
            simulated_date = raw.get("simulated_date")
            event_type = raw.get("event_type", "price_tick")
            
            # Extract tick data dictionary
            tick_dict = raw.get("data") if isinstance(raw.get("data"), dict) else raw
            
            is_valid, err = DataValidator.validate_tick(tick_dict)
            if not is_valid:
                logger.debug(f"Skipping invalid tick: {err}")
                return None
            
            close = float(tick_dict.get("close", tick_dict.get("price", 0.0)))
            open_val = float(tick_dict.get("open", close))
            high_val = float(tick_dict.get("high", max(open_val, close)))
            low_val = float(tick_dict.get("low", min(open_val, close)))
            vol_val = float(tick_dict.get("volume", 0.0))
            ticker = str(tick_dict.get("ticker", "")).upper()
            
            return MarketTick(
                ticker=ticker,
                open=open_val,
                high=high_val,
                low=low_val,
                close=close,
                volume=vol_val,
                simulated_date=simulated_date or tick_dict.get("date"),
                timestamp=timestamp,
                event_type=event_type
            )
        except Exception as e:
            logger.warning(f"Error normalizing tick event: {e}")
            return None

    @classmethod
    def normalize_news_event(cls, raw: dict[str, Any]) -> Optional[NewsItem]:
        """Normalize a news_flash event."""
        try:
            timestamp = raw.get("timestamp")
            simulated_date = raw.get("simulated_date")
            news_dict = raw.get("data") if isinstance(raw.get("data"), dict) else raw
            
            is_valid, err = DataValidator.validate_news(news_dict)
            if not is_valid:
                logger.debug(f"Skipping invalid news: {err}")
                return None
                
            ticker = news_dict.get("ticker")
            if ticker:
                ticker = str(ticker).upper()
                
            return NewsItem(
                ticker=ticker,
                headline=str(news_dict.get("headline", "")).strip(),
                description=str(news_dict.get("description", "")).strip(),
                date=news_dict.get("date") or simulated_date,
                simulated_date=simulated_date,
                timestamp=timestamp
            )
        except Exception as e:
            logger.warning(f"Error normalizing news event: {e}")
            return None
