# sheets_utils.py
import json
import os
from typing import List

import gspread
from google.oauth2.service_account import Credentials

# Import from your config file
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
        # 1. Check if variable exists
        if not GOOGLE_SERVICE_ACCOUNT_JSON:
            raise RuntimeError("CRITICAL: GOOGLE_SERVICE_ACCOUNT_JSON is empty in config/env.")
        
        # 2. Attempt to parse the JSON string (This is where Railway usually fails)
        try:
            info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON Decode Error: Your Railway Variable 'GOOGLE_SERVICE_ACCOUNT_JSON' is broken. It likely has newlines. Please Minify it! Error: {e}")
        
        # 3. Attempt to authorize with Google
        try:
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
            _gc = gspread.authorize(creds)
        except Exception as e:
            raise RuntimeError(f"Google Auth Error: Credentials valid, but login failed. Error: {e}")

    return _gc


def append_ultra_row(values: List[object]) -> None:
    """Append a row to the Ultra 5m sheet."""
    try:
        gc = _get_client()
        sh = gc.open_by_key(ULTRA_SPREADSHEET_ID)
        ws = sh.worksheet(ULTRA_SHEET_NAME)
        ws.append_row(values, value_input_option="RAW")
        print(f"[sheets_utils] Ultra Row Appended Successfully.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"[sheets_utils] ERROR: Worksheet '{ULTRA_SHEET_NAME}' not found. Check tab name.")
    except gspread.exceptions.APIError as e:
        print(f"[sheets_utils] API ERROR: Permission issue? Did you share the sheet with the bot email? {e}")
    except Exception as e:
        print(f"[sheets_utils] Failed to append ultra row: {type(e).__name__} - {e}")


def append_pretty_row(values: List[object]) -> None:
    """Append a row to the Pretty 15m sheet."""
    try:
        gc = _get_client()
        sh = gc.open_by_key(PRETTY_SPREADSHEET_ID)
        ws = sh.worksheet(PRETTY_SHEET_NAME)
        ws.append_row(values, value_input_option="RAW")
        print(f"[sheets_utils] Pretty Row Appended Successfully.")
    except gspread.exceptions.WorksheetNotFound:
        print(f"[sheets_utils] ERROR: Worksheet '{PRETTY_SHEET_NAME}' not found. Check tab name.")
    except gspread.exceptions.APIError as e:
        print(f"[sheets_utils] API ERROR: Permission issue? Did you share the sheet with the bot email? {e}")
    except Exception as e:
        print(f"[sheets_utils] Failed to append pretty row: {type(e).__name__} - {e}")