import logging
import asyncio
from typing import Dict, List, Optional, Any
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from datetime import datetime

from ..config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_PAIRS, TIMEFRAMES

logger = logging.getLogger(__name__)


class BinanceAPI:
    """
    Client for interacting with the Binance API to fetch market data.
    """
    
    def __init__(self):
        """Initialize the Binance API client with API credentials."""
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        self.trading_pairs = TRADING_PAIRS
        self.timeframes = TIMEFRAMES
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # Cache duration in seconds
        
        # Test connection
        try:
            self.client.ping()
            logger.info("Successfully connected to Binance API")
        except BinanceAPIException as e:
            logger.error(f"Failed to connect to Binance API: {e}")
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            The current price as a float
        """
        cache_key = f"price_{symbol}"
        if cache_key in self.cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            
            # Cache the result
            self.cache[cache_key] = price
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return price
        except BinanceAPIException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0
    
    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical klines (candlestick data) for a trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            interval: The time interval (e.g., '1m', '5m', '1h')
            limit: The number of klines to fetch (default: 100)
            
        Returns:
            A pandas DataFrame containing the kline data
        """
        cache_key = f"klines_{symbol}_{interval}_{limit}"
        if cache_key in self.cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Cache the result
            self.cache[cache_key] = df
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return df
        except BinanceAPIException as e:
            logger.error(f"Error fetching klines for {symbol} ({interval}): {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, List]:
        """
        Get the current order book for a trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            limit: The number of orders to fetch (default: 10)
            
        Returns:
            A dictionary containing the bids and asks
        """
        cache_key = f"orderbook_{symbol}_{limit}"
        if cache_key in self.cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            
            # Format the result
            result = {
                'bids': [[float(price), float(qty)] for price, qty in depth['bids']],
                'asks': [[float(price), float(qty)] for price, qty in depth['asks']]
            }
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return result
        except BinanceAPIException as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    def get_all_current_prices(self) -> Dict[str, float]:
        """
        Get the current prices for all trading pairs.
        
        Returns:
            A dictionary mapping trading pair symbols to their current prices
        """
        cache_key = "all_prices"
        if cache_key in self.cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            tickers = self.client.get_all_tickers()
            prices = {ticker['symbol']: float(ticker['price']) for ticker in tickers 
                     if ticker['symbol'] in self.trading_pairs}
            
            # Cache the result
            self.cache[cache_key] = prices
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return prices
        except BinanceAPIException as e:
            logger.error(f"Error fetching all prices: {e}")
            return {}
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market data for a trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            A dictionary containing various market data
        """
        cache_key = f"market_data_{symbol}"
        if cache_key in self.cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.cache[cache_key]
        
        try:
            # Get current price
            price = self.get_current_price(symbol)
            
            # Get 24h ticker
            ticker_24h = self.client.get_ticker(symbol=symbol)
            
            # Get recent trades
            trades = self.client.get_recent_trades(symbol=symbol, limit=10)
            
            # Get order book
            order_book = self.get_order_book(symbol)
            
            # Compile the data
            data = {
                'symbol': symbol,
                'price': price,
                'price_change_24h': float(ticker_24h['priceChange']),
                'price_change_percent_24h': float(ticker_24h['priceChangePercent']),
                'high_24h': float(ticker_24h['highPrice']),
                'low_24h': float(ticker_24h['lowPrice']),
                'volume_24h': float(ticker_24h['volume']),
                'trades': [
                    {
                        'id': trade['id'],
                        'price': float(trade['price']),
                        'qty': float(trade['qty']),
                        'time': datetime.fromtimestamp(trade['time'] / 1000).isoformat(),
                        'is_buyer_maker': trade['isBuyerMaker']
                    }
                    for trade in trades
                ],
                'order_book': order_book,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = data
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return data
        except BinanceAPIException as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_all_market_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive market data for all trading pairs.
        
        Returns:
            A dictionary mapping trading pair symbols to their market data
        """
        result = {}
        for symbol in self.trading_pairs:
            result[symbol] = self.get_market_data(symbol)
        return result
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        self.cache_expiry = {} 