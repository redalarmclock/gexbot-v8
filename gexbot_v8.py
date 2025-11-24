# gexbot_v8.py
import time
import schedule
from collections import deque
from typing import Deque, List, Tuple

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

# --- Constants & Tuning ---

# The total Delta Change required over the window to trigger a signal.
# Backtested optimal value: 1200 BTC.
FLOW_DELTA_REQ = 1200.0 

# Number of snapshots to analyze for the trend (5 * 5m = 25m window).
SLOPE_WINDOW = 5

# Dynamic Slope Threshold (Delta per Step)
# This ensures that if we change the window size, the sensitivity scales correctly.
SLOPE_THRESHOLD = FLOW_DELTA_REQ / SLOPE_WINDOW

HISTORY_LEN = 12
HEARTBEAT_MINS = 60  # Resend message every 60 mins if state hasn't changed

# --- State Management ---
history: Deque[GexSnapshot] = deque(maxlen=HISTORY_LEN)
last_flow_state: str = "Init"
last_msg_ts: float = 0.0

# --- Trend Math ---

def _calculate_slope(values: List[float]) -> float:
    """Calculates Linear Regression Slope (Least Squares)"""
    n = len(values)
    if n < 2: return 0.0
    
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    
    numerator = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    denominator = sum((xs[i] - mean_x) ** 2 for i in range(n))
    
    return numerator / denominator if denominator != 0 else 0.0

def _get_confidence_stars(gamma_structure: float) -> str:
    """
    Returns star rating based on Structural Gamma Magnitude (Billions USD).
    We use Structural Gamma (not Net) because it represents the true Regime strength.
    """
    billions = abs(gamma_structure) / 1_000_000_000
    if billions < 5: return "★☆☆☆☆"
    if billions < 15: return "★★☆☆☆"
    if billions < 30: return "★★★☆☆"
    if billions < 50: return "★★★★☆"
    return "★★★★★"

# --- Logic Engine ---

def compute_directional_bias(history_window: Deque[GexSnapshot], current: GexSnapshot) -> Tuple[str, str]:
    # Need enough data points for the slope window
    if len(history_window) < SLOPE_WINDOW: 
        return "CALIB", "🌊 Flow: Calibrating (gathering history)..."
    
    # Analyze the specific window for trend
    window = list(history_window)[-SLOPE_WINDOW:]
    deltas = [s.net_delta for s in window]
    
    # Calculate Slope
    delta_slope = _calculate_slope(deltas)
    
    # Determine Trend based on Slope vs Dynamic Threshold
    delta_trend = "flat"
    if delta_slope > SLOPE_THRESHOLD: delta_trend = "up"
    elif delta_slope < -SLOPE_THRESHOLD: delta_trend = "down"

    # Tactical Gamma Check (0DTE Risk)
    # If >5B is expiring in <24h, we flag high tactical risk.
    tactical_risk = abs(current.gamma_tactical) > 5_000_000_000 
    
    env = current.gamma_env 
    bias = "Neutral"
    desc = "Chop / No clear edge"
    emoji = "⚖️"
    state_label = "NEUTRAL"

    # --- Logic Matrix ---
    if env == "short":
        if delta_trend == "down":
            bias = "UP (Squeeze)"
            desc = "Dealers chasing upside."
            emoji = "🚀"
            state_label = "UP_SQUEEZE"
        elif delta_trend == "up":
            bias = "DOWN (Flush)"
            desc = "Dealers selling bounces."
            emoji = "🩸"
            state_label = "DOWN_FLUSH"
        else:
            bias = "Neutral (Vol)"
            desc = "High vol chop, balanced flows."
            emoji = "⚡"
            state_label = "NEUTRAL"
            
    elif env == "long":
        if delta_trend == "down":
            bias = "Capped UP"
            desc = "Dealers selling rips (Damping)."
            emoji = "🐌"
            state_label = "CAPPED_UP"
        elif delta_trend == "up":
            bias = "Floored DOWN"
            desc = "Dealers buying dips (Damping)."
            emoji = "🛡️"
            state_label = "FLOORED_DOWN"
        else:
            bias = "Pinned"
            desc = "Mean reversion dominates."
            emoji = "📎"
            state_label = "NEUTRAL"

    # Add Context Tags
    # PATCH: Now uses gamma_structure for stable confidence rating
    confidence = _get_confidence_stars(current.gamma_structure)
    
    extras = []
    if tactical_risk: extras.append("⚠️ 0DTE Vol Risk")
    if current.wide_range_flag: extras.append("🌊 Wide Range")
    
    extra_str = f" | {' '.join(extras)}" if extras else ""
    
    formatted_text = (
        f"{emoji} Flow: {bias}{extra_str}\n"
        f"   💪 Conf: {confidence}\n"
        f"   👉 {desc}"
    )
    
    return state_label, formatted_text

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
    global last_flow_state, last_msg_ts
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        history.append(snapshot)

        base_text = format_pretty(snapshot)
        current_state, flow_line = compute_directional_bias(history, snapshot)
        
        # --- Heartbeat Logic ---
        now = time.time()
        time_since_last = now - last_msg_ts
        is_heartbeat = time_since_last > (HEARTBEAT_MINS * 60)
        
        # Send if State Changed OR Heartbeat expired
        if current_state != last_flow_state or is_heartbeat:
            
            # Add a small header if it's just a heartbeat
            prefix = ""
            if current_state == last_flow_state and is_heartbeat:
                prefix = "🔄 [Update] Regime Unchanged:\n"
            
            final_text = f"{prefix}{base_text}\n\n{flow_line}"
            send_message(final_text)
            
            print(f"[PRETTY] Sent ({current_state})")
            last_flow_state = current_state
            last_msg_ts = now
        else:
            print(f"[PRETTY] Suppressed (State: {current_state})")
        
        # Log Logic
        row_data = snapshot_to_pretty_row(snapshot)
        row_data['flow_bias'] = flow_line
        log_pretty(row_data)
        
    except Exception as e:
        print(f"[PRETTY] Error: {e}")

def main() -> None:
    print("Starting GEX Bot v8.3 (Final Patched Logic)...")
    schedule.every(ULTRA_INTERVAL_MIN).minutes.do(ultra_job)
    schedule.every(PRETTY_INTERVAL_MIN).minutes.do(pretty_job)
    
    # Init
    ultra_job()
    time.sleep(2)
    pretty_job()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()