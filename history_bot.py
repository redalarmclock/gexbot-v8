# history_bot.py
import csv
import os
from datetime import datetime
from typing import Dict, Any
from config import HISTORY_ULTRA_FILE, HISTORY_PRETTY_FILE

# Import sheets_utils if available
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
    """Log a single ultra 5-min print to CSV + Google Sheets."""
    # 1. The new 11-column header (Must match your Google Sheet)
    header = [
        "timestamp_iso",
        "spot",
        "gamma_env",         
        "expiry_outlook",    
        "gamma_tactical",    
        "gamma_structure",   
        "net_gamma",         
        "delta",             
        "magnet",
        "near_range",
        "strong_range"
    ]
    _ensure_file_header(HISTORY_ULTRA_FILE, header)

    timestamp = datetime.utcnow().isoformat()

    # 2. Map the incoming dictionary to the list in the correct order
    csv_values = [
        timestamp,
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
    ]

    # 3. Write to local CSV
    with open(HISTORY_ULTRA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_values)

    # 4. Push to Google Sheets
    if append_ultra_row:
        append_ultra_row(csv_values)

def log_pretty(row: Dict[str, Any]) -> None:
    """Log a pretty print (simplified for readability)."""
    header = ["timestamp_iso", "spot", "gamma_env", "expiry_outlook", "net_gamma", "comment"]
    _ensure_file_header(HISTORY_PRETTY_FILE, header)
    
    csv_values = [
        datetime.utcnow().isoformat(),
        row.get("spot"),
        row.get("gamma_env"),
        row.get("expiry_outlook"),
        row.get("net_gamma"),
        row.get("comment"),
    ]
    
    with open(HISTORY_PRETTY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_values)
        
    if append_pretty_row:
        append_pretty_row(csv_values)