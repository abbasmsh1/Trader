import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Trading configuration
TRADING_PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
STARTING_CAPITAL = 20.0  # USD
TARGET_CAPITAL = 100.0  # USD
TRADING_FEE = 0.001  # 0.1% per trade

# Simulation configuration
SIMULATION_INTERVAL = 60  # seconds between simulation steps
COMMUNICATION_INTERVAL = 300  # seconds between trader communications (5 minutes)
WEBSOCKET_UPDATE_INTERVAL = 1.0  # seconds between WebSocket updates

# Timeframes for analysis
TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    "RSI": {"window": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BOLLINGER": {"window": 20, "std_dev": 2}
}

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True 