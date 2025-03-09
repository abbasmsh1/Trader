import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import time

from .market import Market
from .trader import Trader
from ..agents.warren_weber import WarrenWeber
from ..agents.sonia_soros import SoniaSoros
from ..agents.pete_scalper import PeteScalper
from ..agents.johnny_bollinger import JohnnyBollinger
from ..agents.lena_lynch import LenaLynch
from ..agents.elon_musk import ElonMusk
from ..agents.jeff_bezos import JeffBezos
from ..agents.warren_buffett import WarrenBuffett
from ..agents.ray_dalio import RayDalio
from ..config import SIMULATION_INTERVAL, COMMUNICATION_INTERVAL, TARGET_CAPITAL

logger = logging.getLogger(__name__)


class Simulation:
    """
    Simulation class for managing the trading simulation.
    """
    
    def __init__(self, market: Market):
        """
        Initialize the simulation.
        
        Args:
            market: The market instance
        """
        self.market = market
        self.traders = []
        self.communications = []
        self.running = False
        self.start_time = None
        self.end_time = None
        self.winner = None
        
        # Initialize traders
        self._init_traders()
    
    def _init_traders(self):
        """Initialize the trader agents."""
        self.traders = [
            WarrenWeber(),
            SoniaSoros(),
            PeteScalper(),
            JohnnyBollinger(),
            LenaLynch(),
            ElonMusk(),
            JeffBezos(),
            WarrenBuffett(),
            RayDalio()
        ]
        logger.info(f"Initialized {len(self.traders)} traders")
    
    async def run(self):
        """Run the simulation."""
        if self.running:
            logger.warning("Simulation is already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        logger.info(f"Starting simulation at {self.start_time}")
        
        # Create tasks for simulation and communication
        simulation_task = asyncio.create_task(self._simulation_loop())
        communication_task = asyncio.create_task(self._communication_loop())
        
        # Wait for both tasks to complete
        await asyncio.gather(simulation_task, communication_task)
    
    async def stop(self):
        """Stop the simulation."""
        if not self.running:
            logger.warning("Simulation is not running")
            return
        
        self.running = False
        self.end_time = datetime.now()
        logger.info(f"Stopping simulation at {self.end_time}")
    
    async def _simulation_loop(self):
        """Main simulation loop."""
        while self.running:
            try:
                # Get current market data
                market_data = self.market.get_current_data()
                
                # Extract current prices
                current_prices = {
                    symbol: data['price'] 
                    for symbol, data in market_data.items()
                }
                
                # Update each trader
                for trader in self.traders:
                    if not trader.active:
                        continue
                    
                    # Check if the trader has reached the goal
                    if trader.check_goal_reached(current_prices):
                        if not self.winner:
                            self.winner = trader
                            logger.info(f"We have a winner! {trader.name} reached the goal first!")
                    
                    # Check if the trader is still active
                    portfolio_value = trader.get_portfolio_value(current_prices)
                    if portfolio_value < 1.0:  # If portfolio value drops below $1, the trader is out
                        trader.active = False
                        logger.info(f"Trader {trader.name} is out of the game with portfolio value ${portfolio_value:.2f}")
                        continue
                    
                    # Analyze the market and make trading decisions
                    analysis = trader.analyze_market(market_data)
                    
                    # Execute trades based on analysis
                    if 'trades' in analysis:
                        for trade in analysis['trades']:
                            if trade['action'] == 'buy':
                                symbol = trade['symbol'].replace('USDT', '')
                                amount_usdt = trade['amount_usdt']
                                price = current_prices[trade['symbol']]
                                trader.buy(symbol, amount_usdt, price)
                            elif trade['action'] == 'sell':
                                symbol = trade['symbol'].replace('USDT', '')
                                amount_crypto = trade['amount_crypto']
                                price = current_prices[trade['symbol']]
                                trader.sell(symbol, amount_crypto, price)
                
                # Check if all traders are inactive or if we have a winner
                active_traders = [t for t in self.traders if t.active]
                if not active_traders or self.winner:
                    logger.info("Simulation complete")
                    await self.stop()
                    break
                
                # Wait for the next simulation step
                await asyncio.sleep(SIMULATION_INTERVAL)
            
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(SIMULATION_INTERVAL)
    
    async def _communication_loop(self):
        """Communication loop for trader interactions."""
        while self.running:
            try:
                # Get current market data
                market_data = self.market.get_current_data()
                
                # Generate messages from traders
                for trader in self.traders:
                    if not trader.active:
                        continue
                    
                    # Generate a message
                    other_traders = [t for t in self.traders if t.id != trader.id]
                    message = trader.generate_message(market_data, other_traders)
                    
                    if message:
                        # Record the message
                        message_obj = {
                            'id': str(random.randint(1000000, 9999999)),
                            'trader_id': trader.id,
                            'trader_name': trader.name,
                            'content': message,
                            'timestamp': datetime.now().isoformat(),
                            'responses': []
                        }
                        self.communications.append(message_obj)
                        trader.messages.append(message_obj)
                        
                        # Get responses from other traders
                        for other_trader in other_traders:
                            if not other_trader.active:
                                continue
                            
                            # Generate a response
                            response = other_trader.respond_to_message(message_obj, market_data)
                            
                            if response:
                                # Record the response
                                response_obj = {
                                    'id': str(random.randint(1000000, 9999999)),
                                    'trader_id': other_trader.id,
                                    'trader_name': other_trader.name,
                                    'content': response,
                                    'timestamp': datetime.now().isoformat(),
                                    'in_response_to': message_obj['id']
                                }
                                message_obj['responses'].append(response_obj)
                                other_trader.messages.append(response_obj)
                
                # Wait for the next communication step
                await asyncio.sleep(COMMUNICATION_INTERVAL)
            
            except Exception as e:
                logger.error(f"Error in communication loop: {e}")
                await asyncio.sleep(COMMUNICATION_INTERVAL)
    
    def get_traders_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all traders.
        
        Returns:
            A list of dictionaries containing information about each trader
        """
        return [trader.get_info() for trader in self.traders]
    
    def get_trader_info(self, trader_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific trader.
        
        Args:
            trader_id: The ID of the trader
            
        Returns:
            A dictionary containing information about the trader, or None if not found
        """
        for trader in self.traders:
            if trader.id == trader_id:
                return trader.get_info()
        return None
    
    def get_recent_communications(self) -> List[Dict[str, Any]]:
        """
        Get recent communications between traders.
        
        Returns:
            A list of dictionaries containing recent communications
        """
        # Return the last 20 communications
        return self.communications[-20:] if len(self.communications) > 20 else self.communications
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Get the current status of the simulation.
        
        Returns:
            A dictionary containing the simulation status
        """
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'winner': self.winner.get_info() if self.winner else None,
            'active_traders': len([t for t in self.traders if t.active]),
            'total_traders': len(self.traders)
        } 