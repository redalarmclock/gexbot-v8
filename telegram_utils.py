# telegram_utils.py
import requests
from typing import Optional
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_message(text: str,
                 chat_id: Optional[str] = None,
                 disable_notification: bool = False) -> None:
    """Send a plain text message to Telegram."""
    payload = {
        "chat_id": chat_id or TELEGRAM_CHAT_ID,
        "text": text,
        "disable_notification": disable_notification,
        "parse_mode": "Markdown"
    }
    resp = requests.post(f"{BASE_URL}/sendMessage", data=payload, timeout=10)
    try:
        resp.raise_for_status()
    except Exception as e:
        print(f"[telegram_utils] Failed to send message: {e} | Response: {resp.text}")
