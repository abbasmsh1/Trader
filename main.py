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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crypto_trader')

# Import the database handler
from db.db_handler import DummyDBHandler, PickleDBHandler

class CryptoTraderApp:
    """Main application class for the Crypto Trader system."""
    
    def __init__(self, config_path=None, testnet=True, demo_mode=True):
        """Initialize the crypto trader application."""
        self.logger = logger
        self.testnet = testnet
        self.config_path = config_path
        self.demo_mode = demo_mode
        
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
    
    def initialize(self):
        """Initialize the trading system."""
        logger.info("Initializing Crypto Trader system")
        
        if self.demo_mode:
            logger.info("Using simulated market data (demo mode)")
            
            # Create placeholder agents for demo mode
            # This would be replaced with actual agent creation in a real system
            self.agents = {
                "system_controller": {
                    "id": "system_controller_1",
                    "name": "System Controller",
                    "type": "system_controller",
                    "status": "active"
                }
            }
            
            return True
        
        # In a real system, we would:
        # 1. Create agent instances based on configuration
        # 2. Set up the system controller
        # 3. Activate all agents
        
        logger.warning("Full system initialization not implemented yet")
        return True
    
    def start(self, run_once=False):
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system already running")
            return False
        
        logger.info("Starting Crypto Trader system")
        self.running = True
        self.start_time = datetime.now()
        
        try:
            # Run either a single cycle or continuous operation
            if run_once:
                self._run_cycle()
            else:
                self._run_continuous()
                
            return True
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            return self.stop()
            
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            self.stop()
            return False
    
    def _run_cycle(self):
        """Run a single trading cycle."""
        logger.info("Running trading cycle")
        self.last_update_time = datetime.now()
        
        try:
            # 1. Fetch market data
            market_data = self._fetch_market_data()
            if not market_data:
                logger.warning("No market data available, skipping cycle")
                return
            
            # 2. Update system controller
            if self.system_controller:
                system_update = self.system_controller.process_update({
                    "market_data": market_data,
                    "agents": self.agents
                })
                if not system_update.get("continue_trading", True):
                    logger.warning("System controller requested trading pause")
                    return
            
            # 3. Update all agents
            agent_updates = {}
            for agent_id, agent in self.agents.items():
                if agent.get("status") == "active":
                    try:
                        update_result = agent["instance"].process_update({
                            "market_data": market_data,
                            "system_state": self.get_status()
                        })
                        agent_updates[agent_id] = update_result
                    except Exception as e:
                        logger.error(f"Error updating agent {agent_id}: {str(e)}")
            
            # 4. Execute any pending trades
            self._execute_pending_trades(agent_updates)
            
            # 5. Update performance metrics
            self._update_performance_metrics()
            
            # 6. Save system state
            self._save_system_state()
            
            logger.info("Trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            raise
    
    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data for configured symbols."""
        market_data = {}
        
        try:
            # In demo mode, generate simulated data
            if self.demo_mode:
                for symbol in self.config.get("symbols", []):
                    # Generate random price movements around a base price
                    base_price = 50000.0 if "BTC" in symbol else 3000.0
                    price = base_price * (1 + (random.random() - 0.5) * 0.02)  # ±2% movement
                    volume = random.randint(100, 1000)
                    
                    market_data[symbol] = {
                        "price": price,
                        "volume": volume,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                # TODO: Implement real market data fetching
                logger.warning("Real market data fetching not implemented yet")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
            
        return market_data
    
    def _execute_pending_trades(self, agent_updates: Dict[str, Any]) -> None:
        """Execute any pending trades from agent updates."""
        for agent_id, update in agent_updates.items():
            if "trades" in update:
                for trade in update["trades"]:
                    try:
                        # TODO: Implement trade execution
                        logger.info(f"Would execute trade: {trade}")
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
            self.db_handler.save_state("system_state", state)
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