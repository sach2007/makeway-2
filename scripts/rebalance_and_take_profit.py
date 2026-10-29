"""Script to trigger automated take-profit and stop-loss reconciliation on the live platform."""
from __future__ import annotations
import logging
from dotenv import load_dotenv

load_dotenv(".env")

from trading_bot.config import Settings
from trading_bot.api.client import TradingAPIClient
from trading_bot.api.schemas import OrderRequest, Portfolio, TradeAction
from trading_bot.risk.exposure import ExposureManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("take_profit")


def main():
    print("=" * 80)
    print("      AUTOMATED PROFIT-TAKING & PORTFOLIO RECONCILIATION ENGINE      ")
    print("=" * 80)

    settings = Settings()
    client = TradingAPIClient(settings)

    # 1. Fetch live portfolio state
    raw_port = client.get_portfolio()
    portfolio = Portfolio(**raw_port)
    clock = client.get_clock()
    sim_date = clock.get("simulated_date", "2026-07-28")

    print(f"Team:             {portfolio.team_name} ({settings.redacted_token()})")
    print(f"Current Wealth:   INR {portfolio.total_wealth:,.2f}")
    print(f"Available Cash:   INR {portfolio.capital_remaining:,.2f}")
    print(f"Realized P&L:     INR {portfolio.realized_pnl:,.2f}")
    print(f"Unrealized P&L:   INR {portfolio.unrealized_pnl:,.2f}")
    print(f"Active Holdings:  {len(portfolio.holdings)}")

    # 2. Run ExposureManager exit evaluation
    manager = ExposureManager(
        per_position_stop_loss_pct=0.05,
        take_profit_pct=0.15,
        trailing_stop_pct=0.05
    )

    exits = manager.check_exits(portfolio)
    print(f"\nExits Identified by Automated Engine: {len(exits)}")
    for ticker, action, qty, reason in exits:
        print(f"  --> {ticker}: {action.value} {qty} shares ({reason})")

    # 3. Execute identified exits
    for ticker, action, qty, reason in exits:
        print(f"\nSubmitting {action.value} order for {qty} {ticker} ({reason})...")
        order = OrderRequest(
            ticker=ticker,
            action=action,
            quantity=qty,
            date=sim_date
        )
        try:
            resp = client.place_order(order)
            print(f"  [FILLED] Order ID: {resp.get('order_id')} | Price: {resp.get('execution_price')} | Status: {resp.get('status')}")
        except Exception as e:
            print(f"  [ERROR] Could not submit order: {e}")

    # 4. Check post-execution portfolio
    print("\n" + "=" * 80)
    print("                    POST-EXECUTION PORTFOLIO STATE")
    print("=" * 80)
    updated_port = client.get_portfolio()
    p_up = Portfolio(**updated_port)
    print(f"Updated Total Wealth:   INR {p_up.total_wealth:,.2f}")
    print(f"Updated Cash Balance:   INR {p_up.capital_remaining:,.2f}")
    print(f"Updated Realized P&L:   INR {p_up.realized_pnl:,.2f}  <--- (LOCKED IN PROFIT!)")
    print(f"Updated Unrealized P&L: INR {p_up.unrealized_pnl:,.2f}")
    print(f"Updated Total P&L:      INR {p_up.total_pnl:,.2f} (+{p_up.total_pnl_pct:.2f}%)")
    print("=" * 80)

    client.close()


if __name__ == "__main__":
    main()
