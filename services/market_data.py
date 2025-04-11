"""
Market data service for fetching live data from Binance API.
"""

import ccxt
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time

class MarketDataService:
    """Service for fetching market data from Binance using ccxt."""
    
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        """Initialize the market data service.
        
        Args:
            api_key: Optional Binance API key
            secret: Optional Binance API secret
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True
        })
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_timeout = 60  # Cache timeout in seconds
        
    def get_ticker(self, symbol: str) -> Dict:
        """Fetch ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC/USDT')
            
        Returns:
            Dict containing ticker data
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            return {}
            
    def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[List]:
        """Fetch OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV data
        """
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return []
            
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book data.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to fetch
            
        Returns:
            Dict containing order book data
        """
        try:
            return self.exchange.fetch_order_book(symbol, limit)
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return {'bids': [], 'asks': []}
            
    def get_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Fetch recent trades.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            
        Returns:
            List of recent trades
        """
        try:
            return self.exchange.fetch_trades(symbol, limit=limit)
        except Exception as e:
            self.logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            return []
            
    def get_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict containing ticker, order book and trades data
        """
        cache_key = f"market_data_{symbol}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_timeout:
                return cached_data
                
        try:
            ticker = self.get_ticker(symbol)
            order_book = self.get_order_book(symbol)
            trades = self.get_trades(symbol)
            
            market_data = {
                'ticker': ticker,
                'order_book': order_book,
                'trades': trades,
                'timestamp': current_time
            }
            
            # Update cache
            self._cache[cache_key] = (market_data, current_time)
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return {}
            
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols.
        
        Returns:
            List of trading symbols
        """
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            self.logger.error(f"Error fetching available symbols: {str(e)}")
            return []
            
    def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and limits.
        
        Returns:
            Dict containing exchange information
        """
        try:
            return self.exchange.fetch_exchange_info()
        except Exception as e:
            self.logger.error(f"Error fetching exchange info: {str(e)}")
            return {}
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """Get funding rate for a symbol (for futures trading)."""
        try:
            return await self.exchange.fetch_funding_rate(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {str(e)}")
            return {}
    
    async def get_balance(self) -> Dict:
        """Get account balance."""
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            return {} 