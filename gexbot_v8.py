import time
import re
import schedule
from collections import deque
from typing import Deque, List, Optional, Tuple, Dict, Any

# Keep your existing config/utils
from config import ULTRA_INTERVAL_MIN, PRETTY_INTERVAL_MIN
from telegram_utils import send_message

# Import core logic
from gex_core import (
    fetch_raw_gex_data,
    compute_gex_snapshot,
    format_pretty,
    GexSnapshot,
    snapshot_to_ultra_row,
    snapshot_to_pretty_row,
)

from history_bot import log_ultra, log_pretty

# ============================================================
# V9 — HARD STRUCTURE BOT (Professional Grade)
# ============================================================

# --- Constants & Tuning ---

# Gamma thresholds (USD) - Tuned for Billions
# Alert if Structural Gamma moves by max(2B, 8% of previous)
GAMMA_STRUCTURAL_THRESHOLD_ABS = 2_000_000_000.0   
GAMMA_STRUCTURAL_THRESHOLD_PCT = 0.08              
GAMMA_TACTICAL_THRESHOLD_ABS = 500_000_000.0       # 0.5B for tactical

# Wall resize sensitivity
# Alert if a wall changes size by >20% AND >$800M (Double Filter)
WALL_SIZE_CHANGE_PCT = 0.20                        
WALL_SIZE_CHANGE_ABS_FLOOR = 800_000_000.0         

# Magnet/Flip movement tolerance (in price dollars)
LEVEL_MOVE_THRESHOLD = 500

# Heartbeat (Minutes)
HEARTBEAT_MINS = 60

# --- State ---
history: Deque[GexSnapshot] = deque(maxlen=24)      
last_msg_ts: float = 0.0
last_snapshot: Optional[GexSnapshot] = None


# ---------------------------
# Parsing Helpers
# ---------------------------

_WALL_ITEM_RE = re.compile(
    r"([\d,]+)\s*\(\s*([+-]?\d+(?:\.\d+)?)M\s*([RS])\s*\)"
)

def _parse_wall_string(wall_str: str) -> List[Dict[str, Any]]:
    """
    Parses formatted walls like: '95,000 (-5997.99M R) | ...'
    Handles negative numbers correctly.
    """
    if not wall_str or wall_str.strip() in {"—", "—/—"}:
        return []

    # Split by pipes
    raw_items = re.split(r"\s*\|\s*", wall_str)
    parsed: List[Dict[str, Any]] = []

    for item in raw_items:
        m = _WALL_ITEM_RE.search(item)
        if not m:
            continue
        strike = int(m.group(1).replace(",", ""))
        size_m = float(m.group(2))                 # can be negative
        size = size_m * 1_000_000.0                # M -> USD
        direction = m.group(3)
        parsed.append({"strike": strike, "size": size, "dir": direction})

    return parsed


def _format_billions(val: float) -> str:
    b = val / 1_000_000_000.0
    return f"{b:+.1f}B"


def _level_exists(x: Optional[float]) -> bool:
    return isinstance(x, (int, float))


# ---------------------------
# Structural Change Engine
# ---------------------------

def _struct_gamma_threshold(prev_struct: float) -> float:
    """
    Hybrid threshold: max(2B, 8% of previous magnitude).
    """
    prev_mag = abs(prev_struct)
    return max(GAMMA_STRUCTURAL_THRESHOLD_ABS, GAMMA_STRUCTURAL_THRESHOLD_PCT * prev_mag)


def analyze_structural_changes(prev: GexSnapshot, curr: GexSnapshot) -> Tuple[bool, List[str]]:
    changes: List[str] = []
    should_alert = False

    # 1) Range boundaries (Board Size)
    if prev.strong_range != curr.strong_range:
        changes.append(f"📐 **Strong Range**: {prev.strong_range} → {curr.strong_range}")
        should_alert = True

    if prev.near_range != curr.near_range:
        changes.append(f"🧭 **Near Range**: {prev.near_range} → {curr.near_range}")
        should_alert = True

    # 2) Gamma Hard Numbers (Fuel)
    diff_struct = curr.gamma_structure - prev.gamma_structure
    struct_thr = _struct_gamma_threshold(prev.gamma_structure)
    if abs(diff_struct) >= struct_thr:
        changes.append(f"🌊 **Struct Γ**: {_format_billions(diff_struct)}")
        should_alert = True

    diff_tact = curr.gamma_tactical - prev.gamma_tactical
    if abs(diff_tact) >= GAMMA_TACTICAL_THRESHOLD_ABS:
        changes.append(f"⚡ **Tactical Γ**: {_format_billions(diff_tact)}")
        should_alert = True

    # 3) Regime Flip (Game State)
    if prev.gamma_env != curr.gamma_env:
        changes.append(f"🔄 **Regime Flip**: {prev.gamma_env.upper()} → {curr.gamma_env.upper()}")
        should_alert = True

    # 4) Top Wall Logic (The King)
    prev_walls = _parse_wall_string(prev.top_walls)
    curr_walls = _parse_wall_string(curr.top_walls)

    if prev_walls and curr_walls:
        prev_top = prev_walls[0]
        curr_top = curr_walls[0]

        # A) Strike moved (King Changed)
        if prev_top["strike"] != curr_top["strike"]:
            changes.append(f"🧱 **Top Wall Moved**: {prev_top['strike']} → {curr_top['strike']}")
            should_alert = True
        else:
            # B) Size changed materially (Reinforcement/Decay)
            prev_sz = abs(prev_top["size"])
            curr_sz = abs(curr_top["size"])

            if prev_sz > 0:
                abs_diff = abs(curr_sz - prev_sz)
                pct_diff = (curr_sz - prev_sz) / prev_sz

                # Double Filter: Must be >20% AND >$800M change
                if abs_diff >= WALL_SIZE_CHANGE_ABS_FLOOR and abs(pct_diff) >= WALL_SIZE_CHANGE_PCT:
                    direction = "Reinforced" if pct_diff > 0 else "Decaying" # Logic works for Magnitude
                    # Note: pct_diff calc on magnitude is simpler for alerts
                    changes.append(
                        f"🧱 **Top Wall {direction}**: {pct_diff:+.0%} "
                        f"({curr_sz/1e9:.1f}B)"
                    )
                    should_alert = True

    # 5) Band Walls (Internal Structure)
    if prev.band_walls != curr.band_walls:
        changes.append("🏗️ **Band Re-shuffle**: Internal levels changed.")
        should_alert = True

    # 6) Magnet / Flip (Gravity Points)
    if _level_exists(prev.magnet) and _level_exists(curr.magnet):
        if abs(curr.magnet - prev.magnet) > LEVEL_MOVE_THRESHOLD:
            changes.append(f"🧲 **Magnet Moved**: {int(prev.magnet)} → {int(curr.magnet)}")
            should_alert = True

    if _level_exists(prev.flip) != _level_exists(curr.flip):
        changes.append(f"🧷 **Flip Changed**: {prev.flip if prev.flip else '—'} → {curr.flip if curr.flip else '—'}")
        should_alert = True
    elif _level_exists(prev.flip) and _level_exists(curr.flip):
        if abs(curr.flip - prev.flip) > LEVEL_MOVE_THRESHOLD:
            changes.append(f"🧷 **Flip Moved**: {int(prev.flip)} → {int(curr.flip)}")
            should_alert = True

    return should_alert, changes


# ---------------------------
# Jobs
# ---------------------------

def ultra_job() -> None:
    """
    Logging only. Silent.
    """
    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)
        history.append(snapshot)

        log_ultra(snapshot_to_ultra_row(snapshot))
        print(f"[ULTRA] {snapshot.now_ms} Logged (Silent)")
    except Exception as e:
        print(f"[ULTRA] Error: {e}")


def pretty_job() -> None:
    global last_msg_ts, last_snapshot

    try:
        raw = fetch_raw_gex_data()
        snapshot = compute_gex_snapshot(raw)

        now = time.time()
        time_since_last = now - last_msg_ts
        is_heartbeat = time_since_last > (HEARTBEAT_MINS * 60)

        should_send = False
        header_prefix = ""
        change_text = ""

        if last_snapshot is None:
            should_send = True
            header_prefix = "🟢 **GEX Bot V9 Initialized**\n"
        else:
            is_change, changes = analyze_structural_changes(last_snapshot, snapshot)

            if is_change:
                should_send = True
                header_prefix = "🚨 **STRUCTURE CHANGE**\n"
                change_text = "\n".join([f"• {c}" for c in changes]) + "\n\n"
            elif is_heartbeat:
                should_send = True
                header_prefix = "⏱️ **Hourly Heartbeat**\n"

        if should_send:
            base_text = format_pretty(snapshot)
            final_text = f"{header_prefix}{change_text}{base_text}"
            send_message(final_text)
            print(f"[PRETTY] Sent. (Heartbeat: {is_heartbeat})")
            last_msg_ts = now
        else:
            print("[PRETTY] Structure Stable. No Alert.")

        # Update state + history + logs
        last_snapshot = snapshot
        history.append(snapshot)
        log_pretty(snapshot_to_pretty_row(snapshot))

    except Exception as e:
        print(f"[PRETTY] Error: {e}")


def main() -> None:
    print("Starting GEX Bot v9 (Hard Structure)...")
    print(f"Triggers: Struct Gamma > 2B/8% | Wall Change > 20% & 0.8B")

    # Run immediately
    ultra_job()
    time.sleep(2)
    pretty_job()

    schedule.every(ULTRA_INTERVAL_MIN).minutes.do(ultra_job)
    schedule.every(PRETTY_INTERVAL_MIN).minutes.do(pretty_job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()