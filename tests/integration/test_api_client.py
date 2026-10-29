"""Integration tests against live Order Gateway API at 192.168.65.157:3002."""
import pytest
from trading_bot.config import Settings
from trading_bot.api.client import TradingAPIClient


@pytest.fixture
def live_client():
    settings = Settings()
    client = TradingAPIClient(settings)
    yield client
    client.close()


def test_live_gateway_connectivity(live_client):
    status = live_client.check_connectivity()
    assert status.get("stage", {}).get("status") == 200
    assert status.get("stocks", {}).get("status") == 200


def test_live_stage_endpoint(live_client):
    stage_data = live_client.get_stage()
    assert "stage" in stage_data
    assert "status" in stage_data
    assert stage_data.get("status") in ("running", "RUNNING", "ready", "staging", "idle")


def test_live_clock_endpoint(live_client):
    clock_data = live_client.get_clock()
    assert "current_day" in clock_data
    assert "total_days" in clock_data


def test_live_stocks_universe(live_client):
    stocks = live_client.get_stocks()
    assert isinstance(stocks, list)
    assert len(stocks) > 0
    first = stocks[0]
    assert "ticker" in first or "symbol" in first


def test_live_history_endpoint(live_client):
    stocks = live_client.get_stocks()
    ticker = stocks[0].get("ticker", stocks[0].get("symbol"))
    history = live_client.get_history(ticker, days=30)
    assert isinstance(history, list)
    assert len(history) > 0
    bar = history[0]
    assert "close" in bar
    assert "date" in bar
