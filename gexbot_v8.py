import time
import math
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

# --- Constants & Tuning (v8.6 Combined) ---

# Adjusted FLOW_DELTA_REQ to 750 (was 1200) to match the shorter window.
FLOW_DELTA_REQ = 750.0 

# Reduced Window from 5 -> 3. 
# We now look at the last ~15 mins (if 5m interval) instead of 25 mins.
SLOPE_WINDOW = 3

BASE_SLOPE_THRESHOLD = FLOW_DELTA_REQ / SLOPE_WINDOW

# v8.5.2: Lowered Momentum to 0.12% (was 0.20%)
# Captures the breakout candle earlier.
PRICE_MOMENTUM_THRESHOLD = 0.0012 

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
last_sent_score: float = 0.0  # <--- NEW: Tracks conviction of last alert

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

def _calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculates Pearson correlation coefficient for Divergence checks."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    sum_sq_diff_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_diff_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_diff_x * sum_sq_diff_y)
    
    return numerator / denominator if denominator != 0 else 0.0

def _get_confidence_stars(gamma_structure: float) -> str:
    billions = abs(gamma_structure) / 1_000_000_000
    if billions < 5: return "★☆☆☆☆"
    if billions < 15: return "★★☆☆☆"
    if billions < 30: return "★★★☆☆"
    if billions < 50: return "★★★★☆"
    return "★★★★★"

# --- v8.5.2 Adaptive Logic Components ---

def _get_effective_threshold(snapshot: GexSnapshot) -> float:
    """
    v8.5.1 Logic Retained:
    - Short Gamma: INVERTED (Small structure = Higher threshold/Noise filter).
    - IV: High/Rising IV increases threshold.
    """
    base = BASE_SLOPE_THRESHOLD
    struct_g = abs(snapshot.gamma_structure)
    env = snapshot.gamma_env
    
    # 1. Base Environment Multiplier
    multiplier = 1.0
    
    if env == "short":
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
    v8.5.2 Price Momentum Check (Faster).
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
    v8.5.2 Emergency flip uses DYNAMIC threshold.
    """
    global _flow_state_internal, _last_flow_change_ts
    
    if _flow_state_internal == "Init":