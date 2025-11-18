# gexbot_v8.py
import time
import schedule

from config import ULTRA_INTERVAL_MIN, PRETTY_INTERVAL_MIN
from telegram_utils import send_message
from history_bot import log_ultra, log_pretty
from gex_core import (
    fetch_raw_gex_data,
    compute_gex_snapshot,
    format_ultra,
    format_pretty,
    snapshot_to_ultra_row,
    snapshot_to_pretty_row,
)

def ultra_job() -> None:
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        text = format_ultra(snapshot)

        # send to Telegram
        send_message(text)

        # history bot log
        log_ultra(snapshot_to_ultra_row(snapshot))

        print("[ULTRA] Sent + logged.")
    except Exception as e:
        print(f"[ULTRA] Error: {e}")

def pretty_job() -> None:
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        text = format_pretty(snapshot)

        # send to Telegram
        send_message(text)

        # history bot log
        log_pretty(snapshot_to_pretty_row(snapshot))

        print("[PRETTY] Sent + logged.")
    except Exception as e:
        print(f"[PRETTY] Error: {e}")

def main() -> None:
    print("Starting GEX Bot v8 scheduler...")

    schedule.every(ULTRA_INTERVAL_MIN).minutes.do(ultra_job)
    schedule.every(PRETTY_INTERVAL_MIN).minutes.do(pretty_job)

    # Run once immediately on start (optional)
    ultra_job()
    pretty_job()

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
