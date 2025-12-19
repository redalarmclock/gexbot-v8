import time
import re
import schedule
from collections import deque
from typing import Deque, List, Optional, Tuple, Dict, Any

# Keep your existing config/utils
from config import ULTRA_INTERVAL_MIN, PRETTY_INTERVAL_MIN
from telegram_utils import send_message

# Import core logic (NO changes needed in gex_core.py)
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

# Gamma thresholds (USD)
GAMMA_STRUCTURAL_THRESHOLD_ABS = 2_000_000_000.0   # 2B absolute
GAMMA_STRUCTURAL_THRESHOLD_PCT = 0.08              # 8% of prior struct gamma magnitude (relative)
GAMMA_TACTICAL_THRESHOLD_ABS = 500_000_000.0       # 0.5B absolute (<=24h gamma can be noisier)

# Wall resize sensitivity
WALL_SIZE_CHANGE_PCT = 0.20                        # 20% magnitude change
WALL_SIZE_CHANGE_ABS_FLOOR = 800_000_000.0         # 0.8B floor to prevent tiny-wall % noise

# Magnet/Flip movement tolerance (in price dollars)
LEVEL_MOVE_THRESHOLD = 500

# Heartbeat
HEARTBEAT_MINS = 60

# --- State ---
history: Deque[GexSnapshot] = deque(maxlen=24)      # ~6 hours if pretty is 15m
last_msg_ts: float = 0.0
last_snapshot: Optional[GexSnapshot] = None


# ---------------------------
# Output Cleanup (Hard Structure Only)
# ---------------------------

def _strip_pretty_noise(text: str) -> str:
    """
    Remove non-structural / interpretive lines from format_pretty output.
    Keeps the Pretty block desk-grade: hard numbers + levels only.
    """
    DROP_PREFIXES = (
        "• Trade zone:",
        "• Trade note:",
        "• Bounce map:",
        "• IV:",
        "• Regime stability:",
    )

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith(DROP_PREFIXES):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ---------------------------
# Parsing Helpers
# ---------------------------

_WALL_ITEM_RE = re.compile(
    r"([\d,]+)\s*\(\s*([+-]?\d+(?:\.\d+)?)M\s*([RS])\s*\)"
)

def _parse_wall_string(wall_str: str) -> List[Dict[str, Any]]:
    """
    Parses formatted walls like:
      '95,000 (-5997.99M R) | 90,000 (-5541.03M R) | 85,000 (-4016.77M R)'
    into structured data:
      [{'strike': 95000, 'size': -5.99799e9, 'dir': 'R'}, ...]
    """
    if not wall_str or wall_str.strip() in {"—", "—/—"}:
        return []

    raw_items = re.split(r"\s*\|\s*", wall_str)
    parsed: List[Dict[str, Any]] = []

    for item in raw_items:
        m = _WALL_ITEM_RE.search(item)
        if not m:
            continue
        strike = int(m.group(1).replace(",", ""))
        size_m = float(m.group(2))                 # can be negative
        size = size_m * 1_000_000.0               # M -> USD
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
    Hybrid threshold: max(absolute floor, % of previous magnitude).
    Prevents 2B being too small in extreme regimes and too large in quiet regimes.
    """
    prev_mag = abs(prev_struct)
    return max(GAMMA_STRUCTURAL_THRESHOLD_ABS, GAMMA_STRUCTURAL_THRESHOLD_PCT * prev_mag)


def analyze_structural_changes(prev: GexSnapshot, curr: GexSnapshot) -> Tuple[bool, List[str]]:
    changes: List[str] = []
    should_alert = False

    # 1) Range boundaries (board size)
    if prev.strong_range != curr.strong_range:
        changes.append(f"📐 **Strong Range Shift**: {prev.strong_range} → {curr.strong_range}")
        should_alert = True

    if prev.near_range != curr.near_range:
        changes.append(f"🧭 **Near Range Shift**: {prev.near_range} → {curr.near_range}")
        should_alert = True

    # 2) Gamma hard numbers (structure + tactical)
    diff_struct = curr.gamma_structure - prev.gamma_structure
    struct_thr = _struct_gamma_threshold(prev.gamma_structure)
    if abs(diff_struct) >= struct_thr:
        changes.append(f"🌊 **Struct Γ**: {_format_billions(diff_struct)} (thr {_format_billions(struct_thr)})")
        should_alert = True

    diff_tact = curr.gamma_tactical - prev.gamma_tactical
    if abs(diff_tact) >= GAMMA_TACTICAL_THRESHOLD_ABS:
        changes.append(f"⚡ **Tactical Γ**: {_format_billions(diff_tact)}")
        should_alert = True

    # 3) Regime flip (game state)
    if prev.gamma_env != curr.gamma_env:
        changes.append(f"🔄 **Regime Flip**: {prev.gamma_env.upper()} → {curr.gamma_env.upper()}")
        should_alert = True

    # 4) Top wall logic (king level + magnitude)
    prev_walls = _parse_wall_string(prev.top_walls)
    curr_walls = _parse_wall_string(curr.top_walls)

    if prev_walls and curr_walls:
        prev_top = prev_walls[0]
        curr_top = curr_walls[0]

        # A) Strike moved
        if prev_top["strike"] != curr_top["strike"]:
            changes.append(f"🧱 **Top Wall Moved**: {prev_top['strike']} → {curr_top['strike']}")
            should_alert = True
        else:
            # B) Size changed materially (use magnitude; your sizes can be negative)
            prev_sz = abs(prev_top["size"])
            curr_sz = abs(curr_top["size"])

            if prev_sz > 0:
                abs_diff = abs(curr_sz - prev_sz)
                pct_diff = (curr_sz - prev_sz) / prev_sz

                if abs_diff >= WALL_SIZE_CHANGE_ABS_FLOOR and abs(pct_diff) >= WALL_SIZE_CHANGE_PCT:
                    direction = "Reinforced" if pct_diff > 0 else "Decaying"
                    changes.append(
                        f"🧱 **Top Wall {direction}**: {pct_diff:+.0%} "
                        f"(≈{curr_sz/1e9:.1f}B, Δ≈{abs_diff/1e9:.1f}B)"
                    )
                    should_alert = True

    # 5) Band walls (internal structure)
    # Your core sorts band walls by strike, so string diff is a meaningful reshuffle signal.
    if prev.band_walls != curr.band_walls:
        changes.append("🏗️ **Band Re-shuffle**: Internal band levels changed.")
        should_alert = True

    # 6) Magnet / Flip (gravity points)
    if _level_exists(prev.magnet) and _level_exists(curr.magnet):
        if abs(curr.magnet - prev.magnet) > LEVEL_MOVE_THRESHOLD:
            changes.append(f"🧲 **Magnet Moved**: {int(prev.magnet)} → {int(curr.magnet)}")
            should_alert = True

    # Flip may appear/disappear or move
    if _level_exists(prev.flip) != _level_exists(curr.flip):
        changes.append(
            f"🧷 **Flip Changed**: "
            f"{prev.flip if prev.flip is not None else '—'} → {curr.flip if curr.flip is not None else '—'}"
        )
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
    Logging only. Ultra is silent to keep Telegram clean.
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
            header_prefix = "🟢 **GEX Bot V9 Initialized (Hard Structure)**\n"
        else:
            is_change, changes = analyze_structural_changes(last_snapshot, snapshot)

            if is_change:
                should_send = True
                header_prefix = "🚨 **STRUCTURE CHANGE**\n"
                change_text = "\n".join([f"• {c}" for c in changes]) + "\n\n"
            elif is_heartbeat:
                should_send = True
                header_prefix = "⏱️ **Hourly Heartbeat** (Structure Stable)\n"

        if should_send:
            base_text = _strip_pretty_noise(format_pretty(snapshot))
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
    print(
        "Triggers: "
        f"Struct Γ >= max({_format_billions(GAMMA_STRUCTURAL_THRESHOLD_ABS)}, {GAMMA_STRUCTURAL_THRESHOLD_PCT:.0%} prev) | "
        f"Tactical Γ >= {_format_billions(GAMMA_TACTICAL_THRESHOLD_ABS)} | "
        f"Top wall resize >= {WALL_SIZE_CHANGE_PCT:.0%} and >= {WALL_SIZE_CHANGE_ABS_FLOOR/1e9:.1f}B | "
        f"Level move > {LEVEL_MOVE_THRESHOLD}"
    )

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
