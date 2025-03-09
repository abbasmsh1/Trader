import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

class MarketDataFetcher:
    """
    Fetches and caches market data from cryptocurrency exchanges.
    """
    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize the market data fetcher.
        
        Args:
            exchange_id (str): ID of the exchange to use (default: 'binance')
        """
        load_dotenv()
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': os.getenv(f'{exchange_id.upper()}_API_KEY'),
            'secret': os.getenv(f'{exchange_id.upper()}_API_SECRET'),
            'enableRateLimit': True
        })
        
        self.cache = {}
        self.last_update = {}
        self.cache_ttl = 300  # 5 minutes
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data (e.g., '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check if we have cached data that's still fresh
            current_time = time.time()
            cache_key = f"{symbol}_{timeframe}"
            
            if (cache_key in self.cache and 
                cache_key in self.last_update and 
                current_time - self.last_update[cache_key] < self.cache_ttl):
                return self.cache[cache_key]
            
            # Fetch new data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self.cache[cache_key] = df
            self.last_update[cache_key] = current_time
            
            return df
            
        except Exception as e:
            print(f"Error fetching OHLCV data for {symbol}: {e}")
            
            # Return empty DataFrame or last cached data if available
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> float:
        """
        Fetch the current price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Current price
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            
            # Try to get the last price from cached OHLCV data
            cache_key = f"{symbol}_1h"
            if cache_key in self.cache and not self.cache[cache_key].empty:
                return self.cache[cache_key]['close'].iloc[-1]
            
            return 0.0
    
    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for multiple symbols.
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Dictionary of current prices (symbol -> price)
        """
        prices = {}
        
        for symbol in symbols:
            price = self.fetch_current_price(symbol)
            if price > 0:
                prices[symbol] = price
                
        return prices
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Add RSI
        try:
            from ta.momentum import RSIIndicator
            rsi = RSIIndicator(result['close'], window=14)
            result['rsi'] = rsi.rsi()
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            
        # Add MACD
        try:
            from ta.trend import MACD
            macd = MACD(result['close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            
        # Add Bollinger Bands
        try:
            from ta.volatility import BollingerBands
            bollinger = BollingerBands(result['close'])
            result['bb_upper'] = bollinger.bollinger_hband()
            result['bb_middle'] = bollinger.bollinger_mavg()
            result['bb_lower'] = bollinger.bollinger_lband()
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            
        return result
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100, with_indicators: bool = True) -> pd.DataFrame:
        """
        Get market data with optional technical indicators.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            limit: Number of candles to fetch
            with_indicators: Whether to include technical indicators
            
        Returns:
            DataFrame with market data
        """
        df = self.fetch_ohlcv(symbol, timeframe, limit)
        
        if with_indicators and not df.empty:
            df = self.calculate_indicators(df)
            
        return df 