#!/usr/bin/env python
"""
Crypto Trader - Demo Setup Script

This script creates a demo configuration for running the Crypto Trader
system in simulation mode with historical data.
"""

import os
import json
import argparse
import logging
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("setup_demo")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Demo configuration
DEMO_CONFIG = {
    "mode": "run",  # run in live mode but with demo data
    "use_testnet": True,
    "base_currency": "USDT",
    "initial_balance": 10000.0,
    "update_interval": 10,  # faster updates for demo
    "log_level": "INFO",
    "is_demo": True,
    "exchanges": {
        "binance": {
            "api_key": "demo",
            "api_secret": "demo",
            "testnet": True
        }
    },
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],
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
                "timeframe": "15m",
                "params": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            },
            {
                "name": "RSIStrategy",
                "enabled": True,
                "symbols": ["SOL/USDT", "XRP/USDT", "ADA/USDT"],
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
            "max_slippage": 0.01,
            "simulate_execution": True
        },
        "data_agent": {
            "name": "DataCollectionAgent",
            "enabled": True,
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "use_demo_data": True,
            "demo_data_path": "data/demo"
        }
    },
    "backtesting": {
        "start_date": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "data_source": "csv",
        "data_directory": "data/historical"
    },
    "demo_settings": {
        "price_volatility": 0.005,
        "order_fill_delay": 5,
        "random_seed": 42,
        "simulate_trading_errors": False
    },
    "system": {
        "save_interval": 60,  # more frequent saves for demo
        "db_path": "data/demo/trader.db",
        "cache_dir": "data/demo/cache"
    }
}


def ensure_directories():
    """Create necessary directories for the demo setup."""
    directories = [
        "data",
        "data/demo",
        "data/historical",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_demo_config(output_path=None):
    """Create a demo configuration file."""
    if output_path is None:
        output_path = os.path.join("config", "demo_config.json")
    
    # Ensure the config directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(DEMO_CONFIG, f, indent=4)
    
    logger.info(f"Demo configuration created at: {output_path}")
    return output_path


def create_demo_data():
    """Create basic demo data files if they don't exist."""
    demo_data_dir = os.path.join("data", "demo")
    os.makedirs(demo_data_dir, exist_ok=True)
    
    # Create a placeholder file to indicate demo data is needed
    placeholder_path = os.path.join(demo_data_dir, "GENERATE_DATA")
    
    with open(placeholder_path, 'w') as f:
        f.write("The system will generate demo data during first run.\n")
        f.write(f"Generated on: {datetime.now()}\n")
    
    logger.info("Demo data placeholder created. The system will generate demo data on first run.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup Crypto Trader Demo')
    
    parser.add_argument('--output', type=str, help='Output path for demo configuration')
    
    return parser.parse_args()


def main():
    """Main function to set up the demo."""
    args = parse_arguments()
    
    logger.info("Setting up Crypto Trader Demo...")
    
    # Create necessary directories
    ensure_directories()
    
    # Create demo configuration
    config_path = create_demo_config(args.output)
    
    # Create demo data placeholders
    create_demo_data()
    
    logger.info("Demo setup complete!")
    logger.info(f"To run the demo, use: python main.py --config {config_path}")
    logger.info("To start the web dashboard, use: cd frontend && python app.py")


if __name__ == "__main__":
    main() 