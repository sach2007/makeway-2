import json
import os
import requests
from kafka import KafkaConsumer

GATEWAY = "${e}"
BROKER = "${t}"
USERNAME = "${r}"

# 1. Bearer access token (copied from Step 1 of the Participant Guide)
# Avoid calling POST /api/v1/auth/login repeatedly to prevent server overload
TOKEN = os.environ.get("BOT_ACCESS_TOKEN", "${i}")
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# 2. Connect Kafka consumer to market data feed (enable_auto_commit=False ensures fresh live ticks on reconnect)
consumer = KafkaConsumer(
    "market.ticks",
    "market.news",
    bootstrap_servers=[BROKER],
    group_id=f"group-{USERNAME}",
    auto_offset_reset="latest",
    enable_auto_commit=False,
    value_deserializer=lambda msg: json.loads(msg.decode("utf-8")),
)

print(f"Connected to Kafka broker at {BROKER}. Listening for events...")

tradable_tickers = None
for message in consumer:
    event = message.value
    event_type = event.get("event_type") or event.get("event")

    if event_type == "preflight_tick":
        # Simulation stock universe is available at pre-flight
        if tradable_tickers is None:
            stocks = requests.get(f"{GATEWAY}/api/v1/stocks", headers=headers, timeout=5)
            stocks.raise_for_status()
            tradable_tickers = {stock["ticker"] for stock in stocks.json()["stocks"]}
            print("Simulation tickers:", sorted(tradable_tickers))

        # Acknowledge pre-flight heartbeat sequence
        requests.post(
            f"{GATEWAY}/api/v1/bot/heartbeat",
            headers=headers,
            json={"seq": event["seq"]},
            timeout=2
        ).raise_for_status()

    elif message.topic == "market.ticks":
        ticker = (event.get("data") or {}).get("ticker")
        print("Market update:", event.get("simulated_date"), ticker, event.get("data"))
        # Use tradable_tickers for strategy setup; submit orders only for this universe