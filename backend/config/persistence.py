"""Configuration settings for data persistence."""

import os
from pathlib import Path

# Base directory for all persistent data
BASE_DATA_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "data"

# Individual data directories
TRADE_DATA_DIR = BASE_DATA_DIR / "trades"
STATE_DATA_DIR = BASE_DATA_DIR / "states"
MARKET_DATA_DIR = BASE_DATA_DIR / "market"
ANALYTICS_DIR = BASE_DATA_DIR / "analytics"

# File paths
TRADES_FILE = TRADE_DATA_DIR / "trades.json"
TRADER_STATES_FILE = STATE_DATA_DIR / "trader_states.json"
MARKET_HISTORY_FILE = MARKET_DATA_DIR / "market_history.json"
SIMULATION_STATE_FILE = STATE_DATA_DIR / "simulation_state.json"
ANALYTICS_FILE = ANALYTICS_DIR / "trading_analytics.json"

# Backup settings
BACKUP_DIR = BASE_DATA_DIR / "backups"
MAX_BACKUPS = 5
BACKUP_INTERVAL_HOURS = 24

# Data retention settings
MAX_TRADE_HISTORY_DAYS = 30
MAX_MARKET_HISTORY_DAYS = 7
MAX_COMMUNICATIONS_HISTORY = 1000

# Compression settings
COMPRESS_OLD_DATA = True
COMPRESSION_THRESHOLD_DAYS = 7

# Create all necessary directories
for directory in [TRADE_DATA_DIR, STATE_DATA_DIR, MARKET_DATA_DIR, ANALYTICS_DIR, BACKUP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
