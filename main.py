#!/usr/bin/env python
"""
Crypto Trader - Main Execution Script

This script starts the cryptocurrency trading system by initializing
the agent hierarchy and running the trading loop.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
import signal
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trader.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
try:
    # Import config.py directly instead of from config package
    import config.config as config_module
    from agents.system_controller import SystemControllerAgent
    from models.wallet import Wallet
except ImportError as e:
    logger.error(f"Import error: {str(e)}")
    logger.error("Unable to import required modules. Please check your installation.")
    sys.exit(1)


class CryptoTrader:
    """Main application class for the Crypto Trader system."""
    
    def __init__(self, args):
        """Initialize the trading system."""
        self.args = args
        self.running = False
        self.system_controller = None
        self.config = None
        self.wallet = None
        self.start_time = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info("Initializing Crypto Trader system...")
        
    def setup(self):
        """Set up the trading system."""
        try:
            # Load configuration
            config_path = self.args.config if self.args.config else None
            self.config = config_module.load_config(config_path)
            
            # Override config with command line arguments
            if self.args.mode:
                self.config['mode'] = self.args.mode
            if self.args.testnet is not None:
                self.config['use_testnet'] = self.args.testnet
                
            # Create wallet
            self.wallet = Wallet(
                initial_balance=self.config.get('initial_balance', 10000.0),
                base_currency=self.config.get('base_currency', 'USDT'),
                name="Main Wallet"
            )
            
            # Create system controller (root agent)
            self.system_controller = SystemControllerAgent(
                name="System Controller",
                description="Main controller for the trading system",
                config=self.config,
                parent_id=None
            )
            
            # Set wallet for system controller
            self.system_controller.set_wallet(self.wallet)
            
            # Initialize system
            self.system_controller.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Setup error: {str(e)}", exc_info=True)
            return False
        
    def start(self):
        """Start the trading system."""
        if not self.setup():
            logger.error("Failed to set up the system. Exiting.")
            return False
        
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"Starting Crypto Trader in {self.config.get('mode', 'run')} mode")
        logger.info(f"Using testnet: {self.config.get('use_testnet', True)}")
        
        # Run once or continuously
        if self.args.once:
            self.run_once()
        else:
            self.run_continuously()
            
        return True
    
    def run_once(self):
        """Run a single cycle of the trading system."""
        try:
            logger.info("Running a single trading cycle...")
            
            # Run the system controller once
            result = self.system_controller.run({})
            
            # Log results
            if result:
                logger.info(f"Cycle completed: {json.dumps(result.get('status', {}))}")
            else:
                logger.warning("Cycle completed with no result")
                
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}", exc_info=True)
        finally:
            self.running = False
            self.shutdown()
    
    def run_continuously(self):
        """Run the trading system continuously."""
        logger.info("Running trading system continuously...")
        
        update_interval = self.config.get('update_interval', 60)  # seconds
        
        try:
            while self.running:
                cycle_start = time.time()
                
                try:
                    # Run the system controller
                    result = self.system_controller.run({})
                    
                    # Log results (summarized)
                    if result and 'status' in result:
                        status = result['status']
                        logger.info(f"Cycle completed: {status.get('status')} - Portfolio value: ${status.get('portfolio_value', 0):.2f}")
                except Exception as e:
                    logger.error(f"Error in trading cycle: {str(e)}", exc_info=True)
                
                # Calculate sleep time to maintain consistent update interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, update_interval - cycle_time)
                
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"Continuous run error: {str(e)}", exc_info=True)
        finally:
            self.running = False
            self.shutdown()
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.shutdown()
    
    def shutdown(self):
        """Shut down the trading system."""
        logger.info("Shutting down Crypto Trader system...")
        
        try:
            if self.system_controller:
                self.system_controller.shutdown()
                
            # Calculate runtime
            if self.start_time:
                runtime = datetime.now() - self.start_time
                logger.info(f"System ran for {runtime}")
                
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Trader - Agent-based cryptocurrency trading system')
    
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--mode', type=str, choices=['run', 'backtest', 'optimize', 'status'], 
                        help='Operation mode (run, backtest, optimize, status)')
    parser.add_argument('--testnet', action='store_true', help='Use testnet for trading')
    parser.add_argument('--once', action='store_true', help='Run a single cycle and exit')
    
    return parser.parse_args()


def run_main():
    """Entry point for the trading system when installed as a package."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and start trader
    trader = CryptoTrader(args)
    
    # Run the system
    if not trader.start():
        logger.error("Failed to start Crypto Trader system")
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and start trader
    trader = CryptoTrader(args)
    
    # Run the system
    if not trader.start():
        logger.error("Failed to start Crypto Trader system")
        sys.exit(1) 