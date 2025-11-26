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

# --- New Flow-Stability Tuning (v8.4) ---
MIN_TIME_BETWEEN_FLOW_FLIPS = 10 * 60      # seconds
REQUIRED_CONSEC_SAME_RAW = 2               # consecutive raw states required for flip
SLOPE_OVERSHOOT_FACTOR = 1.2               # emergency flip if slope > 1.2 * threshold
NOISE_GATE_WINDOW = 6                       # how many recent slopes to consider for std
NOISE_GATE_K = 0.7                          # slope must exceed 0.7 * std to be trusted

# --- State Management ---
history: Deque[GexSnapshot] = deque(maxlen=HISTORY_LEN)
last_flow_state: str = "Init"      # Last state used for messaging / heartbeat
last_msg_ts: float = 0.0

# Internal flow engine state (independent from heartbeat state)
_flow_state_internal: str = "Init"
_last_flow_change_ts: float = 0.0
_flow_raw_history: Deque[str] = deque(maxlen=HISTORY_LEN)
_delta_slope_history: Deque[float] = deque(maxlen=HISTORY_LEN)

# --- Trend Math ---

def _calculate_slope(values: List[float]) -> float:
    """Calculates Linear Regression Slope (Least Squares)"""
    n = len(values)
    if n < 2:
        return 0.0
    
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

def _classify_raw_state(env: str, delta_trend: str) -> str:
    """
    Map (gamma_env, delta_trend) -> raw flow state label.
    These are internal labels; presentation is handled separately.
    """
    if env == "short":
        if delta_trend == "down":
            return "UP_SQUEEZE"
        elif delta_trend == "up":
            return "DOWN_FLUSH"
        else:
            return "NEUTRAL_VOL"
    elif env == "long":
        if delta_trend == "down":
            return "CAPPED_UP"
        elif delta_trend == "up":
            return "FLOORED_DOWN"
        else:
            return "PINNED"
    else:
        # flat env → flows less meaningful; treat all as neutral
        return "NEUTRAL_VOL"

def _apply_noise_gate(delta_slope: float) -> bool:
    """
    Returns True if the given slope is strong enough to be trusted,
    False if it should be treated as noise (and thus neutralised).
    """
    _delta_slope_history.append(delta_slope)
    if len(_delta_slope_history) < NOISE_GATE_WINDOW:
        # Not enough history to compute a robust std; accept the signal.
        return True
    
    # Simple std dev of recent slopes
    window = list(_delta_slope_history)[-NOISE_GATE_WINDOW:]
    mean = sum(window) / len(window)
    var = sum((x - mean) ** 2 for x in window) / len(window)
    std = var ** 0.5
    
    if std == 0:
        return True  # no variation; accept
    return abs(delta_slope) >= NOISE_GATE_K * std

def _apply_hysteresis(raw_state: str, delta_slope: float, now: float) -> str:
    """
    Apply time + consistency hysteresis to prevent twitchy flow flips.
    """
    global _flow_state_internal, _last_flow_change_ts
    
    # Initialise on first call
    if _flow_state_internal == "Init":
        _flow_state_internal = raw_state
        _last_flow_change_ts = now
        return _flow_state_internal
    
    # Emergency flip on very strong slope
    if abs(delta_slope) > SLOPE_OVERSHOOT_FACTOR * SLOPE_THRESHOLD:
        _flow_state_internal = raw_state
        _last_flow_change_ts = now
        return _flow_state_internal
    
    # Normal flip: require both time + consecutive raw confirmations
    time_since_change = now - _last_flow_change_ts
    _flow_raw_history.append(raw_state)
    
    if raw_state != _flow_state_internal:
        if time_since_change > MIN_TIME_BETWEEN_FLOW_FLIPS:
            # Check for REQUIRED_CONSEC_SAME_RAW at the tail
            if len(_flow_raw_history) >= REQUIRED_CONSEC_SAME_RAW:
                tail = list(_flow_raw_history)[-REQUIRED_CONSEC_SAME_RAW:]
                if all(s == raw_state for s in tail):
                    _flow_state_internal = raw_state
                    _last_flow_change_ts = now
    
    return _flow_state_internal

def _present_flow(env: str, state_label: str) -> Tuple[str, str, str]:
    """
    Given gamma_env and final state label, return:
    - display_bias (e.g. "UP (Squeeze)")
    - emoji
    - description
    """
    # Defaults
    display_bias = "Neutral (Vol)"
    emoji = "⚖️"
    desc = "Chop / No clear edge"
    
    if env == "short":
        if state_label == "UP_SQUEEZE":
            display_bias = "UP (Squeeze)"
            emoji = "🚀"
            desc = "Dealers chasing upside."
        elif state_label == "DOWN_FLUSH":
            display_bias = "DOWN (Flush)"
            emoji = "🩸"
            desc = "Dealers selling bounces."
        else:
            display_bias = "Neutral (Vol)"
            emoji = "⚡"
            desc = "High vol chop, balanced flows."
    elif env == "long":
        if state_label == "CAPPED_UP":
            display_bias = "Capped UP"
            emoji = "🐌"
            desc = "Dealers selling rips (Damping)."
        elif state_label == "FLOORED_DOWN":
            display_bias = "Floored DOWN"
            emoji = "🛡️"
            desc = "Dealers buying dips (Damping)."
        else:
            display_bias = "Pinned"
            emoji = "📎"
            desc = "Mean reversion dominates."
    else:
        display_bias = "Neutral (Vol)"
        emoji = "⚖️"
        desc = "Gamma flat — flows less directional."
    
    return display_bias, emoji, desc

# --- Logic Engine ---

def compute_directional_bias(history_window: Deque[GexSnapshot], current: GexSnapshot) -> Tuple[str, str]:
    """
    Compute directional flow bias with hysteresis and noise-gating.

    Returns:
        state_label: internal state code used for heartbeat / logging.
        formatted_text: human-readable multi-line string.
    """
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
    if delta_slope > SLOPE_THRESHOLD:
        delta_trend = "up"
    elif delta_slope < -SLOPE_THRESHOLD:
        delta_trend = "down"
    
    # Tactical Gamma Check (0DTE Risk)
    # If >5B is expiring in <24h, we flag high tactical risk.
    tactical_risk = abs(current.gamma_tactical) > 5_000_000_000 
    
    env = current.gamma_env 
    
    # --- Raw State Classification ---
    raw_state = _classify_raw_state(env, delta_trend)
    
    # --- Noise Gate ---
    if not _apply_noise_gate(delta_slope):
        raw_state = "NEUTRAL_VOL"
    
    # --- Hysteresis ---
    now = time.time()
    final_state = _apply_hysteresis(raw_state, delta_slope, now)
    
    # --- Presentation Mapping ---
    bias, emoji, desc = _present_flow(env, final_state)
    
    # Add Context Tags
    confidence = _get_confidence_stars(current.gamma_structure)
    
    extras = []
    if tactical_risk: 
        extras.append("⚠️ 0DTE Vol Risk")
    if current.wide_range_flag: 
        extras.append("🌊 Wide Range")
    
    extra_str = f" | {' '.join(extras)}" if extras else ""
    
    formatted_text = (
        f"{emoji} Flow: {bias}{extra_str}\n"
        f"   💪 Conf: {confidence}\n"
        f"   👉 {desc}"
    )
    
    return final_state, formatted_text

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
    print("Starting GEX Bot v8.4 (Stabilised Flow Logic)...")
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
