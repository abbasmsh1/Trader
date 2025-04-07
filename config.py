"""
Configuration settings for the Crypto Trader application.
This file defines all configurable parameters for the application.
"""
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading pairs to monitor - default list of crypto trading pairs
DEFAULT_SYMBOLS: List[str] = [
    'BTC/USDT',  # Bitcoin
    'ETH/USDT',  # Ethereum
    'BNB/USDT',  # Binance Coin
    'SOL/USDT',  # Solana
    'ADA/USDT',  # Cardano
    'DOGE/USDT'  # Dogecoin
    'XRP/USDT', # Ripple
    'AVAX/USDT', # Avalanche
    'DOT/USDT', # Polkadot
    'MATIC/USDT', # Polygon
    'TRX/USDT', # TRON
    'LTC/USDT', # Litecoin
    'UNI/USDT', # Uniswap
    'LINK/USDT',
    'INJ/USDT', # Injective (AI/Narrative)
    'FET/USDT', # Fetch.ai (AI)
    'RNDR/USDT', # Render (AI/Graphics)
    'GRT/USDT', # The Graph (Indexing)
    'AAVE/USDT', # Aave (DeFi Lending)
    'SUI/USDT', # Sui (New Layer 1)\
    'ARB/USDT', # Arbitrum (Layer 2)
    'OP/USDT', # Optimism (Layer 2)
    'DYDX/USDT', # dYdX (Decentralized Exchange)# Chainlink
    'NEAR/USDT' # Near Protocol
    'EGLD/USDT', # MultiversX (ex-Elrond)
    'HBAR/USDT', # Hedera Hashgraph
    'XTZ/USDT', # Tezos
    'STX/USDT', # Stacks (Bitcoin Layer 2)
    'EDU/USDT', # Open Campus (Education/Web3)
    'ID/USDT', # Space ID (Web3 Domains/Identity)
    'HOOK/USDT', # Hooked Protocol (Web3 Education)
    'ALT/USDT', # AltLayer (Modular Rollups)
    'PORTAL/USDT', # Portal (Web3 Gaming/Infrastructure)
    'PIXEL/USDT', # Pixels (Social/Web3 Game)
    'ACE/USDT', # Fusionist (Web3 Game)
    'NFP/USDT', # Not Financial Advice (Meme/SocialFi)
    'DEGEN/USDT', # Degen (Base Chain / Meme-ish)
    'TNSR/USDT' # Tensor (Solana NFT Marketplace)
]

# Timeframes available for analysis
TIMEFRAMES: Dict[str, str] = {
    '1m': '1 minute',
    '5m': '5 minutes',
    '15m': '15 minutes',
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day'
}

# Default timeframe for initial analysis
DEFAULT_TIMEFRAME: str = '1h'

# Risk levels for trading strategies
RISK_LEVELS: Dict[str, float] = {
    'conservative': 0.3,  # Lower risk, smaller position sizes
    'moderate': 0.6,      # Balanced approach
    'aggressive': 0.9     # Higher risk, larger position sizes
}

# Technical analysis parameters
TECHNICAL_PARAMS: Dict[str, Any] = {
    # Moving averages
    'short_sma': 20,
    'medium_sma': 50,
    'long_sma': 200,
    
    # RSI settings
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    
    # MACD settings
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    # Bollinger Bands
    'bb_period': 20,
    'bb_std_dev': 2
}

# System parameters
SYSTEM_PARAMS: Dict[str, Any] = {
    'update_interval': 60,        # Market data update interval in seconds
    'save_interval': 3600,        # State saving interval in seconds
    'learning_interval': 86400,   # AI model training interval in seconds (daily)
    'max_signals_history': 1000,  # Maximum number of trading signals to store
    'dashboard_refresh_rate': 15000,  # Dashboard refresh rate in milliseconds
    'cache_ttl': 300,             # Cache time-to-live in seconds
    'initial_balance': 10000,     # Initial balance in USDT for simulated trading
    'log_level': 'INFO'           # Logging level
}

# API configuration
API_CONFIG: Dict[str, Any] = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', ''),
        'use_testnet': os.getenv('USE_TESTNET', 'true').lower() == 'true',
        'rate_limit': True,       # Enable rate limiting to avoid API bans
    },
    'fallback_to_simulation': True  # Use simulated data if API fails
}

# Dashboard configuration
DASHBOARD_CONFIG: Dict[str, Any] = {
    'theme': 'dark',              # Dashboard theme (dark/light)
    'chart_height': 500,          # Default chart height in pixels
    'max_displayed_symbols': 6,   # Maximum number of symbols to display
    'default_chart_type': 'candlestick',  # Default chart type
    'default_layout': '2x2',      # Default chart layout
    'port': int(os.getenv('DASHBOARD_PORT', 8050)),  # Dashboard port
    'debug': os.getenv('DEBUG', 'false').lower() == 'true'  # Debug mode
}

# Database configuration
DB_CONFIG: Dict[str, Any] = {
    'type': 'sqlite',
    'path': os.getenv('DB_PATH', 'crypto_trader.db'),
    'backup_interval': 86400  # Daily backup
}

# Trading strategies configuration
STRATEGIES_CONFIG: Dict[str, Dict[str, Any]] = {
    'trend_follower': {
        'enabled': True,
        'risk_level': 'moderate',
        'preferred_timeframe': '4h',
        'parameters': {
            'trend_strength_threshold': 0.5,
            'confirmation_period': 3
        }
    },
    'swing_trader': {
        'enabled': True,
        'risk_level': 'moderate',
        'preferred_timeframe': '1d',
        'parameters': {
            'overbought_threshold': 80,
            'oversold_threshold': 20
        }
    },
    'breakout_trader': {
        'enabled': True,
        'risk_level': 'aggressive',
        'preferred_timeframe': '1h',
        'parameters': {
            'lookback_period': 20,
            'breakout_threshold': 2.0
        }
    },
    'ml_trader': {
        'enabled': True,
        'risk_level': 'moderate',
        'preferred_timeframe': '1h',
        'parameters': {
            'training_period': 90,  # days
            'prediction_horizon': 24,  # hours
            'confidence_threshold': 0.7
        }
    }
} 