# gexbot_v8.py
import time
import schedule
from collections import deque
from typing import Deque, List

from config import ULTRA_INTERVAL_MIN, PRETTY_INTERVAL_MIN
from telegram_utils import send_message

# Import core logic
from gex_core import (
    fetch_raw_gex_data, 
    compute_gex_snapshot, 
    format_ultra, 
    format_pretty, 
    GexSnapshot,
    snapshot_to_ultra_row,
    snapshot_to_pretty_row
)
from history_bot import log_ultra, log_pretty 

# --- History Buffer ---
HISTORY_LEN = 12
history: Deque[GexSnapshot] = deque(maxlen=HISTORY_LEN)

# --- Trend Helpers ---

def _trend_label(values: List[float], eps_abs: float) -> str:
    if len(values) < 2: return "flat"
    delta = values[-1] - values[0]
    if delta > eps_abs: return "up"
    if delta < -eps_abs: return "down"
    return "flat"

def compute_directional_bias(history_window: Deque[GexSnapshot], current: GexSnapshot) -> str:
    if len(history_window) < 3: return "🌊 Flow: Calibrating..."
    
    window = list(history_window)[-3:]
    deltas = [s.net_delta for s in window]
    gammas = [s.net_gamma for s in window]

    # Thresholds
    delta_trend = _trend_label(deltas, eps_abs=500.0) 
    gamma_trend = _trend_label(gammas, eps_abs=1_000_000_000.0)

    env = current.gamma_env 
    bias = "Neutral"
    desc = "Chop / No clear edge"
    emoji = "⚖️"

    if env == "short":
        if delta_trend == "down":
            bias = "UP (Squeeze)"
            desc = "Dealers chasing upside; Squeeze risk High."
            emoji = "🚀"
        elif delta_trend == "up":
            bias = "DOWN (Flush)"
            desc = "Dealers selling bounces; Flush risk High."
            emoji = "🩸"
        else:
            bias = "Neutral (Vol)"
            desc = "High vol chop, but flows are balanced."
            emoji = "⚡"
    elif env == "long":
        if delta_trend == "down":
            bias = "Capped UP"
            desc = "Grind up, but dealers selling into rips."
            emoji = "🐌"
        elif delta_trend == "up":
            bias = "Floored DOWN"
            desc = "Grind down, but dealers buying the dip."
            emoji = "🛡️"
        else:
            bias = "Pinned"
            desc = "Mean reversion dominates."
            emoji = "📎"
    else:
        bias = "Flow-Driven"
        desc = "Spot & Perp flows dominate (GEX Neutral)."
        emoji = "🎲"

    risk_suffix = ""
    if gamma_trend == "down": risk_suffix = " | ⚠️ Risk: Rising"
    elif gamma_trend == "up": risk_suffix = " | 🧊 Risk: Fading"

    return f"{emoji} Flow: {bias}{risk_suffix} -> {desc}"

# --- Jobs ---

def ultra_job() -> None:
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        history.append(snapshot)
        
        text = format_ultra(snapshot)
        send_message(text)
        
        log_ultra(snapshot_to_ultra_row(snapshot))
        print("[ULTRA] Sent + Logged")
    except Exception as e:
        print(f"[ULTRA] Error: {e}")

def pretty_job() -> None:
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        history.append(snapshot)

        base_text = format_pretty(snapshot)
        flow_line = compute_directional_bias(history, snapshot)
        
        final_text = f"{base_text}\n\n{flow_line}"
        send_message(final_text)
        
        # --- NEW: Inject Flow into the Log ---
        row_data = snapshot_to_pretty_row(snapshot)
        row_data['flow_bias'] = flow_line  # Add the calculated flow text
        
        log_pretty(row_data)
        print("[PRETTY] Sent + Logged")
    except Exception as e:
        print(f"[PRETTY] Error: {e}")

def main() -> None:
    print("Starting GEX Bot v8 (Merged Logic)...")
    schedule.every(ULTRA_INTERVAL_MIN).minutes.do(ultra_job)
    schedule.every(PRETTY_INTERVAL_MIN).minutes.do(pretty_job)
    ultra_job()
    time.sleep(2)
    pretty_job()
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()