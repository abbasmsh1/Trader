"""
Configuration management for the Crypto Trader system.

This module handles loading, validating, and providing access to
system configuration settings from JSON files and environment variables.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Logger setup
logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# Default configuration settings
DEFAULT_CONFIG = {
    "mode": "run",  # run, backtest, optimize, status
    "use_testnet": True,
    "base_currency": "USDT",
    "initial_balance": 10000.0,
    "update_interval": 60,  # seconds
    "log_level": "INFO",
    "exchanges": {
        "binance": {
            "api_key": "",
            "api_secret": "",
            "testnet": True
        }
    },
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "agents": {
        "system_controller": {
            "name": "SystemController",
            "enabled": True
        },
        "portfolio_manager": {
            "name": "PortfolioManager",
            "enabled": True,
            "risk_per_trade": 0.02,
            "max_open_trades": 5,
            "allocation_strategy": "equal_weight"
        },
        "strategy_agents": [
            {
                "name": "MACrossStrategy",
                "enabled": True,
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "1h",
                "params": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            },
            {
                "name": "RSIStrategy",
                "enabled": True,
                "symbols": ["SOL/USDT", "XRP/USDT"],
                "timeframe": "15m",
                "params": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                }
            }
        ],
        "execution_agent": {
            "name": "ExecutionAgent",
            "enabled": True,
            "max_slippage": 0.01
        },
        "data_agent": {
            "name": "DataCollectionAgent",
            "enabled": True,
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
        }
    },
    "backtesting": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "data_source": "csv",
        "data_directory": "data/historical"
    },
    "optimization": {
        "method": "grid",
        "metric": "sharpe_ratio",
        "max_runs": 100
    },
    "system": {
        "save_interval": 300,  # seconds
        "db_path": "data/trader.db",
        "cache_dir": "data/cache"
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, with fallback to default settings.
    
    Args:
        config_path: Path to JSON configuration file (optional)
    
    Returns:
        Dict containing configuration settings
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Try to load config from file
    file_config = {}
    
    try:
        # Use provided path or default
        path = config_path or DEFAULT_CONFIG_PATH
        
        if os.path.exists(path):
            logger.info(f"Loading configuration from {path}")
            with open(path, 'r') as f:
                file_config = json.load(f)
            
            # Deep merge with default config
            _deep_merge(config, file_config)
        else:
            logger.warning(f"Configuration file {path} not found, using defaults")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
    
    # Override with environment variables if defined
    _override_from_env(config)
    
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """
    Deep merge two dictionaries, modifying base in place.
    
    Args:
        base: Base dictionary to merge into
        override: Dictionary with values to override
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _override_from_env(config: Dict[str, Any]) -> None:
    """
    Override configuration from environment variables.
    
    Environment variables should be prefixed with CRYPTO_TRADER_
    
    Args:
        config: Configuration dictionary to modify
    """
    # API keys
    if 'CRYPTO_TRADER_BINANCE_API_KEY' in os.environ:
        config['exchanges']['binance']['api_key'] = os.environ['CRYPTO_TRADER_BINANCE_API_KEY']
    
    if 'CRYPTO_TRADER_BINANCE_API_SECRET' in os.environ:
        config['exchanges']['binance']['api_secret'] = os.environ['CRYPTO_TRADER_BINANCE_API_SECRET']
    
    # Other common settings
    if 'CRYPTO_TRADER_MODE' in os.environ:
        config['mode'] = os.environ['CRYPTO_TRADER_MODE']
    
    if 'CRYPTO_TRADER_USE_TESTNET' in os.environ:
        config['use_testnet'] = os.environ['CRYPTO_TRADER_USE_TESTNET'].lower() in ('true', 'yes', '1')
    
    if 'CRYPTO_TRADER_INITIAL_BALANCE' in os.environ:
        try:
            config['initial_balance'] = float(os.environ['CRYPTO_TRADER_INITIAL_BALANCE'])
        except ValueError:
            logger.warning("Invalid value for CRYPTO_TRADER_INITIAL_BALANCE")


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use provided path or default
        path = config_path or DEFAULT_CONFIG_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Remove sensitive data
        save_config = config.copy()
        if 'exchanges' in save_config:
            for exchange in save_config['exchanges'].values():
                if 'api_secret' in exchange:
                    exchange['api_secret'] = ''  # Don't save API secrets to file
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(save_config, f, indent=4)
        
        logger.info(f"Configuration saved to {path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False


def create_default_config(config_path: Optional[str] = None) -> bool:
    """
    Create a default configuration file if it doesn't exist.
    
    Args:
        config_path: Path to save the configuration file (optional)
    
    Returns:
        True if file was created, False if file already exists or an error occurred
    """
    path = config_path or DEFAULT_CONFIG_PATH
    
    if os.path.exists(path):
        logger.info(f"Configuration file {path} already exists")
        return False
    
    return save_config(DEFAULT_CONFIG, path)


if __name__ == "__main__":
    # If run directly, create default config file
    logging.basicConfig(level=logging.INFO)
    created = create_default_config()
    if created:
        print(f"Default configuration created at {DEFAULT_CONFIG_PATH}")
    else:
        print(f"Configuration file already exists at {DEFAULT_CONFIG_PATH}") 