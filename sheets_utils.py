# sheets_utils.py
import json
from typing import List

import gspread
from google.oauth2.service_account import Credentials

from config import (
    ULTRA_SPREADSHEET_ID,
    PRETTY_SPREADSHEET_ID,
    ULTRA_SHEET_NAME,
    PRETTY_SHEET_NAME,
    GOOGLE_SERVICE_ACCOUNT_JSON,
)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_gc = None  # global cached client


def _get_client() -> gspread.Client:
    global _gc
    if _gc is None:
        if not GOOGLE_SERVICE_ACCOUNT_JSON:
            raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")
        info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        _gc = gspread.authorize(creds)
    return _gc


def append_ultra_row(values: List[object]) -> None:
    """Append a row to the Ultra 5m sheet."""
    try:
        gc = _get_client()
        sh = gc.open_by_key(ULTRA_SPREADSHEET_ID)
        ws = sh.worksheet(ULTRA_SHEET_NAME)
        ws.append_row(values, value_input_option="RAW")
    except Exception as e:
        print(f"[sheets_utils] Failed to append ultra row: {e}")


def append_pretty_row(values: List[object]) -> None:
    """Append a row to the Pretty 15m sheet."""
    try:
        gc = _get_client()
        sh = gc.open_by_key(PRETTY_SPREADSHEET_ID)
        ws = sh.worksheet(PRETTY_SHEET_NAME)
        ws.append_row(values, value_input_option="RAW")
    except Exception as e:
        print(f"[sheets_utils] Failed to append pretty row: {e}")
