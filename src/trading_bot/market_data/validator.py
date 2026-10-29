"""Market data validator for tick and news event payloads."""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates raw incoming events from Kafka and REST APIs."""
    
    @staticmethod
    def validate_tick(data: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate price tick data fields.
        Returns (is_valid, error_reason).
        """
        if not isinstance(data, dict):
            return False, "Payload is not a dictionary"
        
        ticker = data.get("ticker")
        if not ticker or not isinstance(ticker, str):
            return False, "Missing or invalid ticker"
        
        close = data.get("close")
        if close is None:
            # Fall back to price if close not present
            close = data.get("price")
            
        if close is None or not isinstance(close, (int, float)) or close <= 0:
            return False, f"Invalid or non-positive close/price: {close}"
        
        # Verify optional numeric fields if present
        for field in ("open", "high", "low"):
            val = data.get(field)
            if val is not None and (not isinstance(val, (int, float)) or val <= 0):
                return False, f"Invalid non-positive {field}: {val}"
                
        volume = data.get("volume")
        if volume is not None and (not isinstance(volume, (int, float)) or volume < 0):
            return False, f"Invalid negative volume: {volume}"
            
        return True, None

    @staticmethod
    def validate_news(data: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate news flash event payload."""
        if not isinstance(data, dict):
            return False, "News payload is not a dictionary"
        
        headline = data.get("headline")
        if not headline or not isinstance(headline, str):
            return False, "Missing or empty headline"
            
        return True, None
