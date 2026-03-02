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


def _ensure_file_header(path: str, header: list) -> None:
    """
    Creates the CSV file with the specified header if it doesn't exist.
    If the file exists, the header is left untouched (data is appended).
    To pick up a new schema, archive/rename the old file first.
    """
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def log_ultra(row: Dict[str, Any]) -> None:
    """
    Ultra Logger (5m): Raw Greeks and structural metrics.
    """
    header = [
        "timestamp_iso", "spot", "gamma_env", "expiry_outlook",
        "gamma_tactical", "gamma_structure", "net_gamma",
        "delta", "magnet", "flip",
        "near_range", "strong_range",
        "intraday_structure",
        "atm_iv", "atm_iv_regime", "atm_iv_trend",
        "regime_stability_score", "regime_stability_label",
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
        row.get("flip"),
        row.get("near_range", ""),
        row.get("strong_range", ""),
        row.get("intraday_structure", ""),
        row.get("atm_iv", ""),
        row.get("atm_iv_regime", ""),
        row.get("atm_iv_trend", ""),
        row.get("regime_stability_score", ""),
        row.get("regime_stability_label", ""),
    ]

    with open(HISTORY_ULTRA_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_values)

    if append_ultra_row:
        append_ultra_row(csv_values)


def log_pretty(row: Dict[str, Any]) -> None:
    """
    Pretty Logger (15m): Full regime log.
    """
    header = [
        "timestamp_iso", "spot", "gamma_env", "expiry_outlook",
        "net_gamma", "delta", "magnet", "flip",
        "top_walls", "band_walls", "band_bias", "band_bias_interpretation",
        "delta_trend", "delta_trend_1h",
        "near_range", "strong_range", "intraday_structure",
        "wide_range_flag",
        "atm_iv", "atm_iv_regime", "atm_iv_trend",
        "regime_stability_score", "regime_stability_label",
    ]
    _ensure_file_header(HISTORY_PRETTY_FILE, header)

    csv_values = [
        datetime.utcnow().isoformat(),
        row.get("spot"),
        row.get("gamma_env"),
        row.get("expiry_outlook"),
        row.get("net_gamma"),
        row.get("delta"),
        row.get("magnet"),
        row.get("flip"),
        row.get("top_walls"),
        row.get("band_walls"),
        row.get("band_bias"),
        row.get("band_bias_interpretation"),
        row.get("delta_trend_short"),
        row.get("delta_trend_long"),
        row.get("near_range", ""),
        row.get("strong_range", ""),
        row.get("intraday_structure", ""),
        row.get("wide_range_flag", ""),
        row.get("atm_iv", ""),
        row.get("atm_iv_regime", ""),
        row.get("atm_iv_trend", ""),
        row.get("regime_stability_score", ""),
        row.get("regime_stability_label", ""),
    ]

    with open(HISTORY_PRETTY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_values)

    if append_pretty_row:
        append_pretty_row(csv_values)
