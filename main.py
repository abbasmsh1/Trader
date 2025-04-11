#!/usr/bin/env python
"""
Multi-Agent AI Trading System with Famous Trader Personas
Main entry point for the trading system.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time
import random
import requests
import hmac
import hashlib
from dotenv import load_dotenv
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_trader')

# Import the database handler
from db.db_handler import DummyDBHandler, PickleDBHandler
from models.wallet import Wallet

class CryptoTraderApp:
    """Main application class for the Crypto Trader system."""
    
    def __init__(self, config_path=None, testnet=True, demo_mode=True):
        """Initialize the crypto trader application."""
        self.logger = logger
        self.testnet = testnet
        self.config_path = config_path
        self.demo_mode = demo_mode
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.models_dir = os.path.join(self.base_dir, 'models')
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.running = False
        self.db_handler = self._setup_database()
        
        # Initialize agent manager
        self.agent_manager = self._setup_agent_manager()
        
        # Register agent types
        self._register_agent_types()
        
        # Initialize agents
        self.agents = {}
        self.system_controller = None
        
        # Performance stats
        self.start_time = None
        self.last_update_time = None
        
        logger.info("CryptoTrader application initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        config = {
            "app_name": "Crypto Trader",
            "version": "0.1.0",
            "testnet": True,
            "demo_mode": True,
            "update_interval": 60,  # seconds
            "initial_balance": 20.0,  # USD/USDT
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "min_trade_value": 5.0,  # Minimum trade value in USD
            "database": {
                "type": "pickle",
                "path": "models"
            },
            "agents": {
                "system_controller": {
                    "name": "System Controller",
                    "enabled": True
                },
                "execution_agent": {
                    "name": "Execution Agent",
                    "enabled": True
                },
                "trader_agents": [
                    {
                        "name": "Buffett Trader",
                        "type": "buffett_trader",
                        "enabled": True,
                        "description": "Value investor focused on fundamentals",
                        "trading_style": "value"
                    },
                    {
                        "name": "Soros Trader",
                        "type": "soros_trader",
                        "enabled": True,
                        "description": "Macro investor focused on reflexivity",
                        "trading_style": "macro"
                    },
                    {
                        "name": "Simons Trader",
                        "type": "simons_trader",
                        "enabled": True,
                        "description": "Quantitative trader using statistical models",
                        "trading_style": "quant"
                    },
                    {
                        "name": "Lynch Trader",
                        "type": "lynch_trader",
                        "enabled": True,
                        "description": "Growth investor looking for tenbaggers",
                        "trading_style": "growth"
                    }
                ]
            }
        }
        
        # Override with file config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge configurations
                    self._deep_update(config, file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return config
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def _setup_database(self):
        """Set up the database handler based on configuration."""
        db_config = self.config.get("database", {})
        db_type = db_config.get("type", "dummy")
        db_path = db_config.get("path", "models")
        
        # Create the database directory if it doesn't exist
        db_dir = os.path.join(self.base_dir, db_path)
        os.makedirs(db_dir, exist_ok=True)
        
        if db_type == "pickle":
            logger.info(f"Using PickleDBHandler with path: {db_dir}")
            return PickleDBHandler(db_dir)
        else:
            logger.info(f"Using DummyDBHandler with path: {db_dir}")
            return DummyDBHandler(db_dir)
    
    def _setup_agent_manager(self):
        """Set up the agent manager."""
        try:
            # Import AgentManager class
            from utils.agent_manager import AgentManager
            return AgentManager(self.db_handler, self.config)
        except ImportError:
            logger.error("Failed to import AgentManager. Creating placeholder.")
            # Create a placeholder agent manager
            class PlaceholderAgentManager:
                def __init__(self, db_handler, config):
                    self.db_handler = db_handler
                    self.config = config
                
                def register_agent_type(self, agent_type, agent_class):
                    logger.info(f"Would register agent type: {agent_type}")
            
            return PlaceholderAgentManager(self.db_handler, self.config)
    
    def _register_agent_types(self):
        """Register all agent types with the agent manager."""
        try:
            # Import agent classes
            # Note: These would be implemented in the agents directory
            from agents.system_controller import SystemControllerAgent
            self.agent_manager.register_agent_type("system_controller", SystemControllerAgent)
            
            # Register other agent types as they become available
            logger.info("Registered agent types: system_controller")
        except ImportError as e:
            logger.warning(f"Could not import agents: {str(e)}")
            logger.info("Running in demo mode with simulated agents")
    
    def _initialize_wallets(self):
        """Initialize wallets for all traders."""
        try:
            # Get list of traders
            traders = self.db_handler.get_traders()
            
            # Create system wallet first
            self.system_wallet = Wallet(trader_id="system")
            self.db_handler.save_wallet(self.system_wallet.to_dict())
            
            # Create wallet for each trader
            for trader in traders:
                wallet = Wallet(trader_id=trader['id'])
                self.db_handler.save_wallet(wallet.to_dict())
            
            logger.info("Wallets initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing wallets: {str(e)}")
            return False
    
    def _load_wallets(self):
        """Load existing wallets from database."""
        try:
            # Get all wallet data
            wallet_data = self.db_handler.get_wallets()
            
            # Create Wallet instances
            self.wallets = {}
            for data in wallet_data:
                wallet = Wallet.from_dict(data)
                self.wallets[wallet.trader_id] = wallet
                
                # Set system wallet reference if found
                if wallet.trader_id == "system":
                    self.system_wallet = wallet
            
            logger.info(f"Loaded {len(self.wallets)} wallets")
            return True
            
        except Exception as e:
            logger.error(f"Error loading wallets: {str(e)}")
            return False
    
    def _initialize_trading_system(self):
        """Initialize the trading system."""
        try:
            # 1. Initialize wallets if not exists
            if not self._load_wallets():
                if not self._initialize_wallets():
                    raise Exception("Failed to initialize wallets")
            
            # 2. Initialize other components
            self._initialize_trading_service()
            self._initialize_performance_tracker()
            
            logger.info("Trading system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            return False
    
    def initialize(self):
        """Initialize the trading system."""
        logger.info("Initializing Crypto Trader system")
        
        try:
            # 1. Initialize wallets
            if not self._load_wallets():
                if not self._initialize_wallets():
                    raise Exception("Failed to initialize wallets")
                if not self._load_wallets():  # Load the newly initialized wallets
                    raise Exception("Failed to load initialized wallets")
            
            # 2. Create and initialize agents
            self._initialize_agents()
            
            # 3. Set up market data service
            self._setup_market_data_service()
            
            # 4. Initialize performance tracking
            self._initialize_performance_tracking()
            
            # 5. Load historical data
            self._load_historical_data()
            
            # 6. Initialize trading service
            self._initialize_trading_service()
            
            logger.info("Crypto Trader system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {str(e)}")
            return False
    
    def _initialize_agents(self):
        """Create and initialize all agents."""
        try:
            # 1. Create system controller
            if self.config.get("agents", {}).get("system_controller", {}).get("enabled", False):
                from agents.system_controller import SystemControllerAgent
                self.system_controller = SystemControllerAgent(
                    name="System Controller",
                    description="Controls overall system operation",
                    config=self.config,
                    wallet=self.system_wallet
                )
                self.agents["system_controller"] = self.system_controller
            
            # 2. Create market analyzers
            for analyzer_config in self.config.get("agents", {}).get("market_analyzer", []):
                if analyzer_config.get("enabled", False):
                    from agents.market_analyzer import MarketAnalyzerAgent
                    analyzer = MarketAnalyzerAgent(
                        name=analyzer_config["name"],
                        description="Analyzes market data",
                        config=analyzer_config,
                        wallet=self.system_wallet
                    )
                    self.agents[analyzer_config["name"]] = analyzer
            
            # 3. Create trader agents
            for trader_config in self.config.get("agents", {}).get("trader_agents", []):
                if trader_config.get("enabled", False):
                    # Import the appropriate trader class based on type
                    trader_type = trader_config.get("type", "base_trader")
                    trader_module = __import__(f"agents.trader.{trader_type}", fromlist=["TraderAgent"])
                    TraderAgent = getattr(trader_module, "TraderAgent")
                    
                    # Create trader instance
                    trader = TraderAgent(
                        name=trader_config["name"],
                        description=trader_config.get("description", ""),
                        config=trader_config,
                        wallet=self.trader_wallets[trader_config["name"]]
                    )
                    self.agents[trader_config["name"]] = trader
            
            # 4. Create execution agent
            if self.config.get("agents", {}).get("execution_agent", {}).get("enabled", False):
                from agents.execution_agent import ExecutionAgent
                self.execution_agent = ExecutionAgent(
                    name="Execution Agent",
                    description="Executes trades in the market",
                    config=self.config.get("agents", {}).get("execution_agent", {}),
                    wallet=self.system_wallet
                )
                self.agents["execution_agent"] = self.execution_agent
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _setup_market_data_service(self):
        """Set up the market data service."""
        try:
            from services.market_data import MarketDataService
            self.market_data_service = MarketDataService(
                config=self.config,
                symbols=self.config.get("symbols", []),
                testnet=self.testnet
            )
            logger.info("Market data service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up market data service: {str(e)}")
            raise
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking system."""
        try:
            from services.performance import PerformanceTracker
            self.performance_tracker = PerformanceTracker(
                config=self.config,
                agents=self.agents,
                wallets=self.trader_wallets
            )
            logger.info("Performance tracking initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing performance tracking: {str(e)}")
            raise
    
    def _load_historical_data(self):
        """Load historical market data."""
        try:
            # Load historical data for each symbol
            for symbol in self.config.get("symbols", []):
                self.market_data_service.load_historical_data(symbol)
            
            logger.info("Historical data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            raise
    
    def _initialize_trading_service(self):
        """Initialize the trading service."""
        try:
            from services.trading import TradingService
            self.trading_service = TradingService(
                trade_execution_service=self.trade_execution_service,
                performance_tracker=self.performance_tracker,
                db_handler=self.db_handler
            )
            logger.info("Trading service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing trading service: {str(e)}")
            raise
    
    def _execute_pending_trades(self, agent_updates: Dict[str, Any]) -> None:
        """Execute any pending trades from agent updates."""
        for agent_id, update in agent_updates.items():
            if "trades" in update:
                for trade in update["trades"]:
                    try:
                        # Create trade signal
                        from services.trading import TradeSignal
                        signal = TradeSignal(
                            trader_id=agent_id,
                            symbol=trade["symbol"],
                            side=trade["side"],
                            order_type=trade["order_type"],
                            amount=Decimal(str(trade["amount"])),
                            price=Decimal(str(trade["price"])) if "price" in trade else None,
                            stop_loss=Decimal(str(trade["stop_loss"])) if "stop_loss" in trade else None,
                            take_profit=Decimal(str(trade["take_profit"])) if "take_profit" in trade else None
                        )
                        
                        # Process the trade
                        result = self.trading_service.process_trade_signal(signal)
                        
                        if result["success"]:
                            logger.info(f"Trade executed successfully for {agent_id}: {result['trade_record']}")
                        else:
                            logger.error(f"Trade execution failed for {agent_id}: {result['error']}")
                            
                    except Exception as e:
                        logger.error(f"Error executing trade for {agent_id}: {str(e)}")
    
    def _update_performance_metrics(self) -> None:
        """Update system performance metrics."""
        # TODO: Implement performance metrics tracking
        pass
    
    def _save_system_state(self) -> None:
        """Save current system state."""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "agents": self.agents,
                "status": self.get_status()
            }
            self.db_handler.save_agent_state("system_state", state)
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
    
    def _run_continuous(self):
        """Run continuous trading operation."""
        logger.info("Starting continuous trading")
        
        # Get update interval from config
        update_interval = self.config.get("update_interval", 60)  # Default to 60 seconds
        
        while self.running:
            try:
                # Run a trading cycle
                self._run_cycle()
                
                # Calculate time until next update
                current_time = datetime.now()
                next_update = self.last_update_time + timedelta(seconds=update_interval)
                sleep_time = (next_update - current_time).total_seconds()
                
                # Sleep until next update, but check for shutdown every second
                while sleep_time > 0 and self.running:
                    time.sleep(min(1.0, sleep_time))
                    sleep_time -= 1.0
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {str(e)}")
                # Sleep for a short time before retrying
                time.sleep(5)
        
        logger.info("Continuous trading stopped")
    
    def stop(self):
        """Stop the trading system."""
        if not self.running:
            logger.warning("Trading system not running")
            return False
        
        logger.info("Stopping Crypto Trader system")
        self.running = False
        
        # Save state before shutting down
        # TODO: Implement state saving
        
        return True
    
    def get_status(self):
        """Get the current status of the trading system."""
        status = {
            "running": self.running,
            "demo_mode": self.demo_mode,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None,
            "agent_count": len(self.agents)
        }
        
        return status


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Crypto Trader - Multi-Agent AI Trading System")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--testnet", "-t", action="store_true", help="Use testnet for live trading")
    parser.add_argument("--demo", "-d", action="store_true", help="Run in demo mode with simulated data")
    parser.add_argument("--run-once", "-o", action="store_true", help="Run a single trading cycle and exit")
    
    args = parser.parse_args()
    
    # Create the trading system
    trading_system = CryptoTraderApp(
        config_path=args.config,
        testnet=args.testnet,
        demo_mode=args.demo if args.demo is not None else True
    )
    
    # Initialize the system
    if not trading_system.initialize():
        logger.error("Failed to initialize trading system")
        return 1
    
    # Start the system
    if not trading_system.start(run_once=args.run_once):
        logger.error("Failed to start trading system")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 