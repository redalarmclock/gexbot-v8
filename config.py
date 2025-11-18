# config.py
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@btcgexbot")

# Schedules
ULTRA_INTERVAL_MIN = 5       # ultra 5-min prints
PRETTY_INTERVAL_MIN = 15     # pretty 15-min prints

# History logging
HISTORY_DIR = "history"
HISTORY_ULTRA_FILE = os.path.join(HISTORY_DIR, "ultra_5m.csv")
HISTORY_PRETTY_FILE = os.path.join(HISTORY_DIR, "pretty_15m.csv")

os.makedirs(HISTORY_DIR, exist_ok=True)

# Safety checks (you’ll see these if env not set)
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID is not set in .env")
