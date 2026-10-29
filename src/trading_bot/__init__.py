"""MANTHAN 2026 Autonomous Algorithmic Trading Bot package."""

from trading_bot.config import Settings, get_settings
from trading_bot.logging_config import setup_logging

__version__ = "0.1.0"
__all__ = ["Settings", "get_settings", "setup_logging"]
