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
    """
    Creates the CSV file with the specified header if it doesn't exist.
    Note: If the file exists but has old headers, new columns will simply be appended
    to the end of new rows. For a clean CSV, rename/archive the old file.
    """
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def log_ultra(row: Dict[str, Any]) -> None:
    """
    Ultra Logger (5m): Focuses on raw Greeks + New v8.4 Metrics.
    """
    header = [
        "timestamp_iso", "spot", "gamma_env", "expiry_outlook",
        "gamma_tactical", "gamma_structure", "net_gamma",
        "delta", "magnet", "near_range", "strong_range",
        # New v8.4 Fields
        "intraday_structure", "atm_iv", "regime_stability_score",
        "trade_score", "trade_zone"
    ]
    _ensure_file_header(HISTORY_ULTRA_FILE, header)

    csv_values = [
        datetime.utcnow().isoformat(),
        row.get("spot"), 
        row.get("gamma_env"), 
        row.get("expiry_outlook"),
        row.get("gamma_tactical"), 
        row.get("gamma_structure"), 
        row.get("net_gamma"),
        row.get("delta"), 
        row.get("magnet"), 
        row.get("near_range"), 
        row.get("strong_range"),
        # New v8.4 Data
        row.get("intraday_structure", ""),
        row.get("atm_iv", ""),
        row.get("regime_stability_score", ""),
        row.get("trade_score", ""),
        row.get("trade_zone", "")
    ]

    with open(HISTORY_ULTRA_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_values)
    
    # Send to Google Sheets if enabled
    if append_ultra_row: 
        append_ultra_row(csv_values)

def log_pretty(row: Dict[str, Any]) -> None:
    """
    Pretty Logger (15m): The Full "Regime" Log.
    Aligned to match the Google Sheet Headers (Columns A -> AA)
    """
    header = [
        "timestamp_iso", "spot", "gamma_env", "expiry_outlook", 
        "flow_bias", "net_gamma", "delta", "magnet", "flip", 
        "walls", "band_walls", "band_bias", "bounce_map", 
        "delta_trend", "delta_trend_1h", "interpretation",
        # --- NEW v8.4 COLUMNS ---
        "near_range", "strong_range", "intraday_structure",
        "atm_iv", "atm_iv_regime", "atm_iv_trend",
        "regime_stability_score", "regime_stability_label",
        "trade_score", "trade_zone", "trade_note"
    ]
    _ensure_file_header(HISTORY_PRETTY_FILE, header)
    
    csv_values = [
        # 1. Standard Identifiers
        datetime.utcnow().isoformat(),
        row.get("spot"),
        row.get("gamma_env"),
        row.get("expiry_outlook"),
        row.get("flow_bias", "Waiting..."), 
        
        # 2. Core Greeks
        row.get("net_gamma"),
        row.get("delta"), # This is 'net_delta' in the dict, but passed as 'delta' in row dict usually. 
                          # CHECK: In gexbot_v8.py, snapshot_to_pretty_row uses 'delta'. 
                          # If gex_core uses 'net_delta', ensure consistency. 
                          # Based on your previous code, it expects 'delta'.
        row.get("magnet"),
        row.get("flip"),
        
        # 3. Structural Features
        row.get("top_walls"),
        row.get("band_walls"),
        row.get("band_bias"),
        row.get("bounce_map"),
        
        # 4. Trends & Interp
        row.get("delta_trend_short"),
        row.get("delta_trend_long"),
        row.get("interpretation"),

        # 5. --- NEW v8.4 COLUMNS (Appended to Right) ---
        row.get("near_range", ""),
        row.get("strong_range", ""),
        row.get("intraday_structure", ""),
        row.get("atm_iv", ""),
        row.get("atm_iv_regime", ""),
        row.get("atm_iv_trend", ""),
        row.get("regime_stability_score", ""),
        row.get("regime_stability_label", ""),
        row.get("trade_score", ""),
        row.get("trade_zone", ""),
        row.get("trade_note", "")
    ]
    
    with open(HISTORY_PRETTY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_values)
        
    # Send to Google Sheets if enabled
    if append_pretty_row:
        append_pretty_row(csv_values)