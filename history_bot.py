# history_bot.py
import csv
import os
from datetime import datetime
from typing import Dict, Any
from config import HISTORY_ULTRA_FILE, HISTORY_PRETTY_FILE

try:
    from sheets_utils import append_ultra_row, append_pretty_row
except ImportError:
    append_ultra_row = None
    append_pretty_row = None

def _ensure_file_header(path: str, header: list[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def log_ultra(row: Dict[str, Any]) -> None:
    header = [
        "timestamp_iso", "spot", "gamma_env", "expiry_outlook",
        "gamma_tactical", "gamma_structure", "net_gamma",
        "delta", "magnet", "near_range", "strong_range"
    ]
    _ensure_file_header(HISTORY_ULTRA_FILE, header)

    csv_values = [
        datetime.utcnow().isoformat(),
        row.get("spot"), row.get("gamma_env"), row.get("expiry_outlook"),
        row.get("gamma_tactical"), row.get("gamma_structure"), row.get("net_gamma"),
        row.get("delta"), row.get("magnet"), row.get("near_range"), row.get("strong_range")
    ]

    with open(HISTORY_ULTRA_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_values)
    if append_ultra_row: append_ultra_row(csv_values)

def log_pretty(row: Dict[str, Any]) -> None:
    """
    Updated Pretty Logger: Captures Flow, Walls, Trends, and Interpretation.
    """
    header = [
        "timestamp_iso", "spot", "gamma_env", "expiry_outlook", 
        "flow_bias", "net_gamma", "delta", "magnet", "flip", 
        "walls", "band_walls", "band_bias", "bounce_map", 
        "delta_trend", "delta_trend_1h", "interpretation"
    ]
    _ensure_file_header(HISTORY_PRETTY_FILE, header)
    
    csv_values = [
        datetime.utcnow().isoformat(),
        row.get("spot"),
        row.get("gamma_env"),
        row.get("expiry_outlook"),
        row.get("flow_bias", "Waiting..."), # The new Flow line
        row.get("net_gamma"),
        row.get("delta"),
        row.get("magnet"),
        row.get("flip"),
        row.get("top_walls"),
        row.get("band_walls"),
        row.get("band_bias"),
        row.get("bounce_map"),
        row.get("delta_trend_short"),
        row.get("delta_trend_long"),
        row.get("interpretation")
    ]
    
    with open(HISTORY_PRETTY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_values)
        
    if append_pretty_row:
        append_pretty_row(csv_values)