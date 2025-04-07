#!/usr/bin/env python
"""
Crypto Trader - Main execution script

This script initializes and runs the trading system with all agents.
It sets up the agent hierarchy, initializes components, and starts the trading loop.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trader.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("crypto_trader")

# Import agent components
from agents.base_agent import BaseAgent
from agents.agent_manager import AgentManager
from agents.system_controller import SystemControllerAgent
from agents.market_analyzer import MarketAnalyzerAgent
from agents.strategy.trend_strategy import TrendStrategyAgent
from agents.strategy.swing_strategy import SwingStrategyAgent
from agents.strategy.breakout_strategy import BreakoutStrategyAgent
from agents.portfolio_manager import PortfolioManagerAgent
from agents.execution_agent import ExecutionAgent
from agents.risk_manager import RiskManagerAgent

# Import config and database
from config.agent_config import create_agent_instances, AGENT_HIERARCHY

class CryptoTraderApp:
    """Main application class for the Crypto Trader system."""
    
    def __init__(self, config_path=None, testnet=True):
        """Initialize the crypto trader application."""
        self.logger = logger
        self.testnet = testnet
        self.config_path = config_path
        self.running = False
        self.db_handler = self._setup_database()
        
        # Initialize agent manager
        self.agent_manager = AgentManager(self.db_handler)
        
        # Register agent types
        self._register_agent_types()
        
        # Initialize agents
        self.agents = {}
        self.system_controller = None
        
        # Performance stats
        self.start_time = None
        self.last_update_time = None
    
    def _setup_database(self):
        """Set up a basic database handler for agent persistence."""
        # This is a placeholder. In a real implementation, you would:
        # 1. Initialize a proper database connection
        # 2. Set up schemas if needed
        # 3. Return a database handler
        
        # For now, return a dummy handler with basic dict-based storage
        class DummyDBHandler:
            def __init__(self):
                self.storage = {}
                
            def save_agent_state(self, agent_id, state):
                self.storage[agent_id] = state
                return True
                
            def load_agent_state(self, agent_id):
                return self.storage.get(agent_id)
                
            def list_agent_states(self):
                return list(self.storage.keys())
        
        return DummyDBHandler()
    
    def _register_agent_types(self):
        """Register all agent types with the agent manager."""
        self.agent_manager.register_agent_type("system_controller", SystemControllerAgent)
        self.agent_manager.register_agent_type("market_analyzer", MarketAnalyzerAgent)
        self.agent_manager.register_agent_type("trend_strategy", TrendStrategyAgent)
        self.agent_manager.register_agent_type("swing_strategy", SwingStrategyAgent)
        self.agent_manager.register_agent_type("breakout_strategy", BreakoutStrategyAgent)
        self.agent_manager.register_agent_type("portfolio_manager", PortfolioManagerAgent)
        self.agent_manager.register_agent_type("execution", ExecutionAgent)
        self.agent_manager.register_agent_type("risk_manager", RiskManagerAgent)
    
    def initialize(self):
        """Initialize the trading system."""
        self.logger.info("Initializing Crypto Trader system")
        
        # Create agent instances based on configuration
        self.agents = create_agent_instances(self.agent_manager, self.db_handler)
        
        # Get system controller reference
        for agent_id, agent in self.agents.items():
            if agent.agent_type == "system_controller":
                self.system_controller = agent
                break
        
        if not self.system_controller:
            self.logger.error("Failed to initialize system controller")
            return False
        
        # Activate all agents
        for agent_id, agent in self.agents.items():
            agent.activate()
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
        return True
    
    def start(self, run_once=False):
        """Start the trading system."""
        if not self.system_controller:
            self.logger.error("Cannot start: System controller not initialized")
            return False
        
        self.logger.info("Starting Crypto Trader system")
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
            self.logger.info("Received shutdown signal")
            return self.stop()
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {str(e)}")
            self.stop()
            return False
    
    def _run_cycle(self):
        """Run a single cycle of the trading system."""
        self.logger.info("Running a single system cycle")
        
        # Get current market data
        market_data = self._fetch_market_data()
        
        # Prepare input for the system controller
        system_input = {
            "market_data": market_data,
            "timestamp": datetime.now().isoformat(),
            "mode": "single_cycle"
        }
        
        # Run the system controller
        result = self.system_controller.run(system_input)
        
        # Log results
        self.logger.info(f"Cycle completed - Status: {result.get('status', 'unknown')}")
        
        return result
    
    def _run_continuous(self):
        """Run the trading system in continuous mode."""
        self.logger.info("Running trading system in continuous mode")
        
        update_interval = self.system_controller.config.get("update_interval", 60)
        
        while self.running:
            cycle_start = time.time()
            
            try:
                result = self._run_cycle()
                self.last_update_time = datetime.now()
                
                # If the system is in emergency shutdown, stop running
                if result.get("status") == "emergency_shutdown":
                    self.logger.warning("Emergency shutdown triggered - stopping system")
                    self.stop()
                    break
                
                # Calculate sleep time to maintain update interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, update_interval - elapsed)
                
                if sleep_time > 0:
                    self.logger.info(f"Waiting {sleep_time:.1f}s until next cycle")
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Error during trading cycle: {str(e)}")
                # Continue running despite errors, the system controller
                # should handle error counts and emergency shutdown if needed
                time.sleep(update_interval)  # Use standard interval on error
    
    def _fetch_market_data(self):
        """Fetch current market data for all required symbols."""
        # In a real implementation, this would connect to exchanges/APIs
        # For this demo, return simulated data
        
        # Get list of required symbols from agents
        required_symbols = set()
        for agent_id, agent in self.agents.items():
            if agent.agent_type == "market_analyzer":
                symbols = agent.config.get("symbols", [])
                for symbol in symbols:
                    required_symbols.add(symbol)
        
        # Generate dummy data for each symbol
        market_data = {}
        for symbol in required_symbols:
            # Simple simulated OHLCV data with some randomness
            import numpy as np
            
            base_price = {
                "BTC/USDT": 50000,
                "ETH/USDT": 3000,
                "BNB/USDT": 400,
                "SOL/USDT": 100,
                "ADA/USDT": 1.2,
                "DOGE/USDT": 0.15
            }.get(symbol, 100)
            
            # Generate some fake candles
            candles = []
            for i in range(30):  # 30 periods of data
                close = base_price * (1 + np.random.normal(0, 0.02))  # 2% volatility
                open_price = close * (1 + np.random.normal(0, 0.01))
                high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
                low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
                volume = base_price * np.random.normal(100, 20)
                
                # [timestamp, open, high, low, close, volume]
                candle = [time.time() - (30-i) * 3600, open_price, high, low, close, volume]
                candles.append(candle)
                
                # Update base price for next candle
                base_price = close
            
            # Generate indicator data
            rsi_values = [50 + np.random.normal(0, 10) for _ in range(30)]
            
            # Create market data entry
            market_data[symbol] = {
                "ohlcv": candles,
                "indicators": {
                    "rsi": rsi_values,
                    "stoch": {
                        "k": [50 + np.random.normal(0, 15) for _ in range(30)],
                        "d": [50 + np.random.normal(0, 10) for _ in range(30)]
                    },
                    "bbands": {
                        "upper": [candles[i][4] * 1.05 for i in range(30)],
                        "middle": [candles[i][4] for i in range(30)],
                        "lower": [candles[i][4] * 0.95 for i in range(30)]
                    },
                    "macd": {
                        "macd": [np.random.normal(0, 5) for _ in range(30)],
                        "signal": [np.random.normal(0, 3) for _ in range(30)],
                        "histogram": [np.random.normal(0, 2) for _ in range(30)]
                    }
                }
            }
        
        return market_data
    
    def stop(self):
        """Stop the trading system gracefully."""
        self.logger.info("Stopping Crypto Trader system")
        self.running = False
        
        # Save state for all agents
        for agent_id, agent in self.agents.items():
            try:
                state = agent.save_state()
                self.db_handler.save_agent_state(agent_id, state)
                agent.deactivate()
            except Exception as e:
                self.logger.error(f"Error saving state for agent {agent_id}: {str(e)}")
        
        self.logger.info("Trading system stopped")
        return True
    
    def get_status(self):
        """Get the current status of the trading system."""
        if not self.system_controller:
            return {
                "status": "not_initialized",
                "message": "Trading system not initialized"
            }
        
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Get status from system controller
        controller_status = self.system_controller.get_status()
        
        return {
            "status": "running" if self.running else "stopped",
            "uptime_seconds": uptime,
            "agent_count": len(self.agents),
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "system_status": controller_status
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Crypto Trader System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", choices=["run", "backtest", "optimize", "status"], default="run",
                       help="Operation mode")
    parser.add_argument("--testnet", action="store_true", default=True,
                       help="Use testnet for trading")
    parser.add_argument("--once", action="store_true", 
                       help="Run a single cycle and exit")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create and initialize the trading system
    trading_system = CryptoTraderApp(config_path=args.config, testnet=args.testnet)
    
    if args.mode == "run":
        # Initialize the system
        if not trading_system.initialize():
            logger.error("Failed to initialize trading system")
            return 1
        
        # Start trading
        trading_system.start(run_once=args.once)
        
    elif args.mode == "backtest":
        logger.info("Backtest mode not implemented yet")
        
    elif args.mode == "optimize":
        logger.info("Optimization mode not implemented yet")
        
    elif args.mode == "status":
        if not trading_system.initialize():
            logger.error("Failed to initialize trading system")
            return 1
            
        status = trading_system.get_status()
        logger.info(f"System status: {status['status']}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 