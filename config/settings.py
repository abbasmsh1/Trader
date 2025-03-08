# Trading pairs to monitor
DEFAULT_SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'ADA/USDT',
    'SOL/USDT'
]

# Timeframes available for analysis
TIMEFRAMES = {
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day'
}

# Risk levels
RISK_LEVELS = {
    'conservative': 0.3,
    'moderate': 0.6,
    'aggressive': 0.8
}

# Technical analysis parameters
TECHNICAL_PARAMS = {
    'short_sma': 20,
    'long_sma': 50,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

# System parameters
SYSTEM_PARAMS = {
    'update_interval': 300,  # 5 minutes in seconds
    'max_signals_history': 1000,
    'dashboard_refresh_rate': 300000  # 5 minutes in milliseconds
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'trading_system.log',
            'mode': 'a',
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 