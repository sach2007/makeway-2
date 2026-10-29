"""Configuration management with validation."""
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


def _load_yaml(filename: str) -> dict:
    config_dir = Path(__file__).parent.parent.parent / "config"
    path = config_dir / filename
    if path.exists():
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Application settings loaded from environment and config files."""
    # API
    api_base_url: str = Field(default="http://192.168.65.157:3002")
    api_token: str = Field(default_factory=lambda: os.environ.get("BOT_ACCESS_TOKEN", os.environ.get("API_TOKEN", "")))
    
    # Kafka
    kafka_bootstrap_servers: str = Field(default="192.168.65.157:9094")
    kafka_group_id: str = Field(default="trading-bot")
    kafka_tick_topic: str = Field(default="market.ticks")
    kafka_news_topic: str = Field(default="market.news")
    
    # Logging
    log_level: str = Field(default="INFO")
    
    # Strategy params (loaded from yaml)
    lookback_short: int = 5
    lookback_medium: int = 20
    lookback_long: int = 50
    signal_threshold: float = 0.3
    no_trade_band: float = 0.1
    cooldown_ticks: int = 3
    momentum_weight: float = 0.35
    trend_weight: float = 0.35
    mean_reversion_weight: float = 0.2
    news_weight: float = 0.1
    
    # Trading
    max_order_retries: int = 3
    order_timeout_seconds: int = 10
    rate_limit_per_second: int = 50
    min_confidence: float = 0.3
    
    # Data
    tick_window_size: int = 100
    history_warmup_days: int = 30
    stale_data_threshold_seconds: int = 60
    
    # Risk
    max_position_pct: float = 0.15
    max_total_long_pct: float = 0.85
    max_total_short_pct: float = 0.30
    max_gross_exposure_pct: float = 1.0
    max_single_trade_pct: float = 0.10
    max_open_positions: int = 15
    max_outstanding_orders: int = 20
    max_drawdown_pct: float = 0.15
    per_position_loss_pct: float = 0.05
    max_consecutive_losses: int = 5
    emergency_halt_drawdown_pct: float = 0.25
    min_cash_reserve_pct: float = 0.05
    
    # Machine Learning
    ml_enabled: bool = True
    models_dir: str = "models"
    ml_weight: float = 0.25
    target_horizon: int = 3
    cost_hurdle_bps: float = 20.0
    
    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
    
    def __init__(self, **kwargs: Any):
        # Load YAML configs and merge
        strategy_cfg = _load_yaml("settings.yaml").get("strategy", {})
        trading_cfg = _load_yaml("settings.yaml").get("trading", {})
        data_cfg = _load_yaml("settings.yaml").get("data", {})
        ml_cfg = _load_yaml("settings.yaml").get("ml", {})
        risk_cfg = _load_yaml("risk_limits.yaml").get("risk", {})
        merged = {**strategy_cfg, **trading_cfg, **data_cfg, **ml_cfg, **risk_cfg, **kwargs}
        super().__init__(**merged)
    
    @field_validator('api_token')
    @classmethod
    def token_not_placeholder(cls, v: str) -> str:
        if v == 'your_bearer_token_here':
            return ''
        return v
    
    def validate_required(self) -> list[str]:
        """Return list of missing required settings."""
        issues = []
        if not self.api_base_url:
            issues.append("API_BASE_URL is required")
        if not self.api_token:
            issues.append("API_TOKEN is required")
        return issues
    
    @property
    def api_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token and self.api_token.strip():
            headers["Authorization"] = f"Bearer {self.api_token.strip()}"
        return headers
    
    def redacted_token(self) -> str:
        if len(self.api_token) > 8:
            return self.api_token[:4] + '****' + self.api_token[-4:]
        return '****'


def get_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
