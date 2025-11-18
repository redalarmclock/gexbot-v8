# history_bot.py

from sheets_utils import append_ultra_row, append_pretty_row
from datetime import datetime


import csv
import os
from datetime import datetime
from typing import Dict, Any

from config import HISTORY_ULTRA_FILE, HISTORY_PRETTY_FILE

def _ensure_file_header(path: str, header: list[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def log_ultra(row: Dict[str, Any]) -> None:
    """Log a single ultra 5-min print to CSV + Google Sheets."""
    header = [
        "timestamp_iso",
        "spot",
        "gamma_sign",
        "gamma_flip",
        "magnet",
        "delta",
        "net_gamma",
        "near_range",
        "strong_range",
    ]
    _ensure_file_header(HISTORY_ULTRA_FILE, header)

    timestamp = datetime.utcnow().isoformat()

    csv_values = [
        timestamp,
        row.get("spot"),
        row.get("gamma_sign"),
        row.get("gamma_flip"),
        row.get("magnet"),
        row.get("delta"),
        row.get("net_gamma"),
        row.get("near_range"),
        row.get("strong_range"),
    ]

    # Write to local CSV
    with open(HISTORY_ULTRA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_values)

    # Also push to Google Sheets
    append_ultra_row(csv_values)

def log_pretty(row: Dict[str, Any]) -> None:
    """Log a single pretty 15-min print to CSV + Google Sheets."""
    header = [
        "timestamp_iso",
        "spot",
        "gamma_env_label",
        "magnet",
        "delta",
        "net_gamma",
        "comment",
    ]
    _ensure_file_header(HISTORY_PRETTY_FILE, header)

    timestamp = datetime.utcnow().isoformat()

    csv_values = [
        timestamp,
        row.get("spot"),
        row.get("gamma_env_label"),
        row.get("magnet"),
        row.get("delta"),
        row.get("net_gamma"),
        row.get("comment"),
    ]

    # Write to local CSV
    with open(HISTORY_PRETTY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_values)

    # Also push to Google Sheets
    append_pretty_row(csv_values)
