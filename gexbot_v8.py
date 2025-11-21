# gexbot_v8.py
import time
import schedule
from collections import deque
from typing import Deque, List

from config import ULTRA_INTERVAL_MIN, PRETTY_INTERVAL_MIN
from telegram_utils import send_message

# We import the correct class and functions from your CURRENT gex_core
from gex_core import (
    fetch_raw_gex_data, 
    compute_gex_snapshot, 
    format_ultra, 
    format_pretty, 
    GexSnapshot,
    snapshot_to_ultra_row,
    snapshot_to_pretty_row
)
# Re-enable logging (ensure sheets_utils is fixed first)
from history_bot import log_ultra, log_pretty 

# --- History Buffer ---
# Keep last 12 snapshots (~1 hour of data)
HISTORY_LEN = 12
history: Deque[GexSnapshot] = deque(maxlen=HISTORY_LEN)

# --- Trend Helpers ---

def _trend_label(values: List[float], eps_abs: float) -> str:
    """Returns 'up', 'down', or 'flat' based on the window."""
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    if delta > eps_abs:
        return "up"
    if delta < -eps_abs:
        return "down"
    return "flat"

def compute_directional_bias(history_window: Deque[GexSnapshot], current: GexSnapshot) -> str:
    """
    Calculates Flow Bias based on Delta trends within the Gamma Environment.
    Uses variable names compatible with your current GexSnapshot.
    """
    if len(history_window) < 3:
        return "🌊 Flow: Calibrating..."

    # Take last 3 snapshots
    window = list(history_window)[-3:]

    # extracting metrics using YOUR current gex_core variable names
    # (net_delta, net_gamma are the correct fields in your God Mode file)
    deltas = [s.net_delta for s in window]
    gammas = [s.net_gamma for s in window]

    # Thresholds (Tuned for BTC)
    # 500 BTC delta change is significant flow
    delta_trend = _trend_label(deltas, eps_abs=500.0) 
    # 1 Billion USD Gamma change is significant structure change
    gamma_trend = _trend_label(gammas, eps_abs=1_000_000_000.0)

    # Get Environment (short/long) from your existing field 'gamma_env'
    env = current.gamma_env 

    bias = "Neutral"
    desc = "Chop / No clear edge"
    emoji = "⚖️"

    # --- LOGIC ENGINE ---
    if env == "short":
        # Short Gamma: Hedging ACCELERATES moves
        if delta_trend == "down":
            # Delta getting more negative -> Dealers must BUY -> Squeeze
            bias = "UP (Squeeze)"
            desc = "Dealers chasing upside; Squeeze risk High."
            emoji = "🚀"
        elif delta_trend == "up":
            # Delta getting more positive -> Dealers must SELL -> Flush
            bias = "DOWN (Flush)"
            desc = "Dealers selling bounces; Flush risk High."
            emoji = "🩸"
        else:
            bias = "Neutral (Vol)"
            desc = "High vol chop, but flows are balanced."
            emoji = "⚡"

    elif env == "long":
        # Long Gamma: Hedging DAMPENS moves
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
        # Neutral / Flat environment
        bias = "Flow-Driven"
        desc = "Spot & Perp flows dominate (GEX Neutral)."
        emoji = "🎲"

    # Momentum Suffix
    risk_suffix = ""
    if gamma_trend == "down":   # Gamma getting more negative (Riskier)
        risk_suffix = " | ⚠️ Risk: Rising"
    elif gamma_trend == "up":   # Gamma getting less negative (Safer)
        risk_suffix = " | 🧊 Risk: Fading"

    return f"{emoji} Flow: {bias}{risk_suffix}\n   👉 {desc}"

# --- Jobs ---

def ultra_job() -> None:
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        history.append(snapshot)
        
        # Ultra doesn't strictly need the bias line, but we send it standard
        text = format_ultra(snapshot)
        send_message(text)
        
        # Logging
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
        
        # Calculate the new Directional Bias (Flow Line)
        flow_line = compute_directional_bias(history, snapshot)
        
        final_text = f"{base_text}\n\n{flow_line}"
        send_message(final_text)
        
        # Logging
        log_pretty(snapshot_to_pretty_row(snapshot))
        print("[PRETTY] Sent + Logged")
    except Exception as e:
        print(f"[PRETTY] Error: {e}")

def main() -> None:
    print("Starting GEX Bot v8 (Merged Logic)...")

    schedule.every(ULTRA_INTERVAL_MIN).minutes.do(ultra_job)
    schedule.every(PRETTY_INTERVAL_MIN).minutes.do(pretty_job)

    # Run once on startup
    ultra_job()
    time.sleep(2)
    pretty_job()

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()