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

# The BASE Delta Change required over the window to trigger a signal.
FLOW_DELTA_REQ = 1200.0 
SLOPE_WINDOW = 5
BASE_SLOPE_THRESHOLD = FLOW_DELTA_REQ / SLOPE_WINDOW

# v8.5.1: Tightened Momentum (0.20% move over window required)
PRICE_MOMENTUM_THRESHOLD = 0.0020 

HISTORY_LEN = 12
HEARTBEAT_MINS = 60  

# Flow Stability Tuning
MIN_TIME_BETWEEN_FLOW_FLIPS = 10 * 60      
REQUIRED_CONSEC_SAME_RAW = 2               
SLOPE_OVERSHOOT_FACTOR = 1.2               
NOISE_GATE_WINDOW = 6                       
NOISE_GATE_K = 0.7                          

# --- State Management ---
history: Deque[GexSnapshot] = deque(maxlen=HISTORY_LEN)
last_flow_state: str = "Init"      
last_msg_ts: float = 0.0

# Internal flow engine state 
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
    billions = abs(gamma_structure) / 1_000_000_000
    if billions < 5: return "★☆☆☆☆"
    if billions < 15: return "★★☆☆☆"
    if billions < 30: return "★★★☆☆"
    if billions < 50: return "★★★★☆"
    return "★★★★★"

# --- v8.5.1 New Logic Components ---

def _get_effective_threshold(snapshot: GexSnapshot) -> float:
    """
    v8.5.1 Adaptive Thresholds (Refined):
    - Short Gamma: Now INVERTED.
        * Small Structure: Higher threshold (Filter noise).
        * Large Structure: Lower threshold (Trust the heavy flow).
    - IV: High/Rising IV increases threshold.
    """
    base = BASE_SLOPE_THRESHOLD
    struct_g = abs(snapshot.gamma_structure)
    env = snapshot.gamma_env
    
    # 1. Base Environment Multiplier
    multiplier = 1.0
    
    if env == "short":
        # GPT/Gemini Consensus:
        # If structure is SMALL (<15B), it's noisy/fragile. Be Conservative.
        if struct_g < 15_000_000_000:
            multiplier = 1.2  # Tighter (Filter chop)
        # If structure is HUGE (>35B), moves are real. Be Sensitive.
        elif struct_g > 35_000_000_000:
            multiplier = 0.8  # Looser (Catch the trend)
        else:
            multiplier = 1.0  # Standard
            
    elif env == "long":
        # Long Gamma dampens vol. We need a strong signal to care.
        multiplier = 1.1
        
    else: # Flat
        # Noise factory. Clamp hard.
        multiplier = 1.6
        
    # 2. Stability & IV Adjustment
    
    # IV Check: High Volatility = High Noise. Require stronger signal.
    if "High" in snapshot.atm_iv_regime or "Rising" in snapshot.atm_iv_trend:
        multiplier *= 1.2
        
    # Stability Check
    if "Stable" in snapshot.regime_stability_label:
        multiplier *= 0.9  # Trust stable regimes more
    elif "Chaotic" in snapshot.regime_stability_label:
        multiplier *= 1.3  # Don't trust chaos
        
    return base * multiplier

def _calculate_price_momentum(window: List[GexSnapshot]) -> str:
    """
    v8.5 Price Momentum Check.
    Returns: "up" / "down" / "flat"
    """
    if not window: return "flat"
    
    prices = [s.spot for s in window]
    start_p = prices[0]
    end_p = prices[-1]
    
    pct_change = (end_p - start_p) / start_p
    
    if pct_change > PRICE_MOMENTUM_THRESHOLD:
        return "up"
    elif pct_change < -PRICE_MOMENTUM_THRESHOLD:
        return "down"
    return "flat"

def _classify_raw_state_v85(env: str, delta_trend: str, price_trend: str) -> str:
    """
    v8.5 Combined Classification:
    Requires Delta AND Price to agree for a directional signal.
    """
    if env == "short":
        # Short Gamma: Dealers Sell Dips / Buy Rips.
        # SQUEEZE: Delta Falling (Buying Futures) + Price Rising.
        if delta_trend == "down" and price_trend == "up":
            return "UP_SQUEEZE"
        # FLUSH: Delta Rising (Selling Futures) + Price Falling.
        elif delta_trend == "up" and price_trend == "down":
            return "DOWN_FLUSH"
        
        else:
            return "NEUTRAL_VOL"

    elif env == "long":
        # Long Gamma: Dealers Sell Rips / Buy Dips.
        if delta_trend == "down": # Buying
            if price_trend == "down": return "FLOORED_DOWN" # Buying into the drop
            return "PINNED" 
        elif delta_trend == "up": # Selling
            if price_trend == "up": return "CAPPED_UP" # Selling into the rally
            return "PINNED"
        else:
            return "PINNED"
            
    else:
        return "NEUTRAL_VOL"

def _apply_noise_gate(delta_slope: float) -> bool:
    _delta_slope_history.append(delta_slope)
    if len(_delta_slope_history) < NOISE_GATE_WINDOW:
        return True
    
    window = list(_delta_slope_history)[-NOISE_GATE_WINDOW:]
    mean = sum(window) / len(window)
    var = sum((x - mean) ** 2 for x in window) / len(window)
    std = var ** 0.5
    
    if std == 0: return True
    return abs(delta_slope) >= NOISE_GATE_K * std

def _apply_hysteresis(raw_state: str, delta_slope: float, current_threshold: float, now: float) -> str:
    """
    v8.5.1 Update: Emergency flip now uses the DYNAMIC threshold, not base.
    """
    global _flow_state_internal, _last_flow_change_ts
    
    if _flow_state_internal == "Init":
        _flow_state_internal = raw_state
        _last_flow_change_ts = now
        return _flow_state_internal
    
    # Emergency flip: Scale based on the EFFECTIVE threshold
    if abs(delta_slope) > SLOPE_OVERSHOOT_FACTOR * current_threshold:
        _flow_state_internal = raw_state
        _last_flow_change_ts = now
        return _flow_state_internal
    
    time_since_change = now - _last_flow_change_ts
    _flow_raw_history.append(raw_state)
    
    if raw_state != _flow_state_internal:
        if time_since_change > MIN_TIME_BETWEEN_FLOW_FLIPS:
            if len(_flow_raw_history) >= REQUIRED_CONSEC_SAME_RAW:
                tail = list(_flow_raw_history)[-REQUIRED_CONSEC_SAME_RAW:]
                if all(s == raw_state for s in tail):
                    _flow_state_internal = raw_state
                    _last_flow_change_ts = now
    
    return _flow_state_internal

def _present_flow(env: str, state_label: str) -> Tuple[str, str, str]:
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
    
    return display_bias, emoji, desc

# --- Logic Engine ---

def compute_directional_bias(history_window: Deque[GexSnapshot], current: GexSnapshot) -> Tuple[str, str]:
    if len(history_window) < SLOPE_WINDOW: 
        return "CALIB", "🌊 Flow: Calibrating (gathering history)..."
    
    window = list(history_window)[-SLOPE_WINDOW:]
    deltas = [s.net_delta for s in window]
    
    # 1. Adaptive Threshold (Uses IV + Structure Inversion)
    eff_threshold = _get_effective_threshold(current)
    
    # 2. Delta Trend
    delta_slope = _calculate_slope(deltas)
    delta_trend = "flat"
    if delta_slope > eff_threshold:
        delta_trend = "up"
    elif delta_slope < -eff_threshold:
        delta_trend = "down"
        
    # 3. Price Momentum (v8.5)
    price_trend = _calculate_price_momentum(window)
    
    env = current.gamma_env 
    
    # 4. Classification
    raw_state = _classify_raw_state_v85(env, delta_trend, price_trend)
    
    # 5. Noise Gate
    if not _apply_noise_gate(delta_slope):
        raw_state = "NEUTRAL_VOL"
    
    # 6. Hysteresis (Passes eff_threshold for emergency logic)
    now = time.time()
    final_state = _apply_hysteresis(raw_state, delta_slope, eff_threshold, now)
    
    # 7. Presentation
    bias, emoji, desc = _present_flow(env, final_state)
    
    confidence = _get_confidence_stars(current.gamma_structure)
    extras = []
    if abs(current.gamma_tactical) > 5_000_000_000: extras.append("⚠️ 0DTE Vol Risk")
    if current.wide_range_flag: extras.append("🌊 Wide Range")
    
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
        
        now = time.time()
        time_since_last = now - last_msg_ts
        is_heartbeat = time_since_last > (HEARTBEAT_MINS * 60)
        
        if current_state != last_flow_state or is_heartbeat:
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
        
        row_data = snapshot_to_pretty_row(snapshot)
        row_data['flow_bias'] = flow_line
        log_pretty(row_data)
        
    except Exception as e:
        print(f"[PRETTY] Error: {e}")

def main() -> None:
    print("Starting GEX Bot v8.5.1 (Refined Adaptive Logic)...")
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