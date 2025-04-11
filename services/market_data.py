"""
Market Data Service - Handles market data operations for the trading system.
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import ccxt
import time

class MarketDataService:
    """Service for handling market data operations."""
    
    def __init__(self, config: Dict[str, Any], symbols: List[str], testnet: bool = True):
        """
        Initialize the market data service.
        
        Args:
            config: Configuration dictionary
            symbols: List of trading symbols
            testnet: Whether to use testnet
        """
        self.config = config
        self.symbols = symbols
        self.testnet = testnet
        self.logger = logging.getLogger('market_data_service')
        
        # Initialize exchange
        self.exchange = self._setup_exchange()
        
        # Initialize data storage
        self.market_data = {}
        self.historical_data = {}
        
        self.logger.info("Market Data Service initialized")
    
    def _setup_exchange(self) -> ccxt.Exchange:
        """Set up the exchange connection."""
        try:
            # Get API credentials from environment variables
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            # Initialize exchange
            exchange_class = getattr(ccxt, 'binance')
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            
            # Set testnet if enabled
            if self.testnet:
                exchange.set_sandbox_mode(True)
            
            self.logger.info(f"Exchange connection established (testnet: {self.testnet})")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Error setting up exchange: {str(e)}")
            raise
    
    def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data for all symbols."""
        try:
            market_data = {}
            
            for symbol in self.symbols:
                try:
                    # Convert symbol to exchange format
                    exchange_symbol = symbol.replace('/', '')
                    
                    # Fetch ticker data
                    ticker = self.exchange.fetch_ticker(exchange_symbol)
                    
                    # Fetch order book
                    order_book = self.exchange.fetch_order_book(exchange_symbol, limit=5)
                    
                    # Fetch recent trades
                    trades = self.exchange.fetch_trades(exchange_symbol, limit=10)
                    
                    # Calculate volume from recent trades
                    volume = sum(float(trade['amount']) for trade in trades)
                    
                    # Store market data
                    market_data[symbol] = {
                        'price': float(ticker['last']),
                        'bid': float(ticker['bid']),
                        'ask': float(ticker['ask']),
                        'high': float(ticker['high']),
                        'low': float(ticker['low']),
                        'volume': volume,
                        'timestamp': ticker['timestamp'],
                        'order_book': {
                            'bids': order_book['bids'],
                            'asks': order_book['asks']
                        }
                    }
                    
                    self.logger.info(f"Fetched market data for {symbol}: {market_data[symbol]['price']}")
                    
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            # Update stored market data
            self.market_data = market_data
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return {}
    
    def load_historical_data(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """
        Load historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame containing historical data
        """
        try:
            # Convert symbol to exchange format
            exchange_symbol = symbol.replace('/', '')
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                exchange_symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Store historical data
            self.historical_data[symbol] = df
            
            self.logger.info(f"Loaded {len(df)} historical data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical data for a symbol within a time range.
        
        Args:
            symbol: Trading symbol
            start_time: Start time for data
            end_time: End time for data
            
        Returns:
            DataFrame containing filtered historical data
        """
        try:
            if symbol not in self.historical_data:
                self.load_historical_data(symbol)
            
            df = self.historical_data[symbol]
            
            # Filter by time range if provided
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            if symbol in self.market_data:
                return self.market_data[symbol]['price']
            
            # If not in stored data, fetch fresh data
            self.fetch_market_data()
            
            if symbol in self.market_data:
                return self.market_data[symbol]['price']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None

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