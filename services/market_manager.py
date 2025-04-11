"""
Market manager service for handling market data updates and distribution.
"""

import asyncio
import logging
from typing import Dict, List, Set, Callable
from datetime import datetime
from .market_data import MarketDataService

class MarketManager:
    """Manager for handling market data updates and distribution."""
    
    def __init__(self, market_data_service: MarketDataService):
        """Initialize the market manager."""
        self.market_data = market_data_service
        self.logger = logging.getLogger(__name__)
        
        # Set of symbols to track
        self._tracked_symbols: Set[str] = set()
        
        # Subscribers for market data updates
        self._subscribers: Dict[str, List[Callable]] = {}
        
        # Update task
        self._update_task = None
        self._running = False
        
        # Update interval (in seconds)
        self._update_interval = 1.0
        
    async def start(self):
        """Start the market data update loop."""
        if self._running:
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Market manager started")
        
    async def stop(self):
        """Stop the market data update loop."""
        if not self._running:
            return
            
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Market manager stopped")
        
    def subscribe(self, symbol: str, callback: Callable):
        """Subscribe to market data updates for a symbol."""
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []
            self._tracked_symbols.add(symbol)
            
        self._subscribers[symbol].append(callback)
        self.logger.info(f"New subscriber for {symbol}")
        
    def unsubscribe(self, symbol: str, callback: Callable):
        """Unsubscribe from market data updates for a symbol."""
        if symbol in self._subscribers:
            self._subscribers[symbol].remove(callback)
            if not self._subscribers[symbol]:
                del self._subscribers[symbol]
                self._tracked_symbols.remove(symbol)
            self.logger.info(f"Subscriber removed for {symbol}")
            
    async def _update_loop(self):
        """Main update loop for market data."""
        while self._running:
            try:
                # Update all tracked symbols
                for symbol in self._tracked_symbols:
                    await self._update_symbol(symbol)
                    
                # Wait for next update
                await asyncio.sleep(self._update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _update_symbol(self, symbol: str):
        """Update market data for a specific symbol."""
        try:
            # Get latest market data
            data = await self.market_data.get_market_data(symbol)
            
            # Notify subscribers
            if symbol in self._subscribers:
                for callback in self._subscribers[symbol]:
                    try:
                        await callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in subscriber callback for {symbol}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error updating {symbol}: {str(e)}")
            
    async def get_historical_data(self, symbol: str, timeframe: str, since: datetime, limit: int = 1000) -> List[Dict]:
        """Get historical market data for a symbol."""
        try:
            # Convert datetime to timestamp
            since_timestamp = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = await self.market_data.get_ohlcv(symbol, timeframe, limit)
            
            # Convert to list of dictionaries
            return [{
                'timestamp': candle[0],
                'open': candle[1],
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5]
            } for candle in ohlcv]
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return []
            
    def set_update_interval(self, interval: float):
        """Set the update interval in seconds."""
        self._update_interval = max(0.1, interval)  # Minimum 0.1 seconds
        self.logger.info(f"Update interval set to {self._update_interval} seconds")
        
    def get_tracked_symbols(self) -> List[str]:
        """Get list of currently tracked symbols."""
        return list(self._tracked_symbols)
        
    def get_subscriber_count(self, symbol: str) -> int:
        """Get number of subscribers for a symbol."""
        return len(self._subscribers.get(symbol, [])) 