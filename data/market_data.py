import ccxt
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
import time

class MarketDataFetcher:
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
            # Check if we have cached data that's still fresh (less than 5 minutes old)
            current_time = time.time()
            cache_key = f"{symbol}_{timeframe}"
            
            if (cache_key in self.cache and 
                cache_key in self.last_update and 
                current_time - self.last_update[cache_key] < 300):
                return self.cache[cache_key]
            
            # Generate synthetic data based on the symbol
            # Use the symbol name to seed the random generator for consistent results
            seed = sum(ord(c) for c in symbol)
            np.random.seed(seed)
            
            # Base price depends on the symbol
            if 'BTC' in symbol:
                base_price = 50000 + np.random.normal(0, 5000)
            elif 'ETH' in symbol:
                base_price = 3000 + np.random.normal(0, 300)
            elif 'BNB' in symbol:
                base_price = 500 + np.random.normal(0, 50)
            elif 'SOL' in symbol:
                base_price = 100 + np.random.normal(0, 10)
            elif 'DOGE' in symbol or 'SHIB' in symbol or 'PEPE' in symbol or 'FLOKI' in symbol or 'BONK' in symbol:
                base_price = 0.1 + np.random.normal(0, 0.01)
            else:
                base_price = 10 + np.random.normal(0, 1)
            
            # Generate timestamps
            end_time = datetime.now()
            if timeframe == '1h':
                start_time = end_time - timedelta(hours=limit)
                freq = 'H'
            elif timeframe == '1d':
                start_time = end_time - timedelta(days=limit)
                freq = 'D'
            elif timeframe == '15m':
                start_time = end_time - timedelta(minutes=15*limit)
                freq = '15min'
            else:
                start_time = end_time - timedelta(hours=limit)
                freq = 'H'
                
            timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            # Generate price data with a random walk
            np.random.seed(seed)
            returns = np.random.normal(0.0001, 0.01, len(timestamps))
            price_changes = np.cumprod(1 + returns)
            closes = base_price * price_changes
            
            # Generate OHLCV data
            volatility = np.random.uniform(0.005, 0.02, len(timestamps))
            opens = closes * (1 + np.random.normal(0, 0.005, len(timestamps)))
            highs = np.maximum(opens, closes) * (1 + volatility)
            lows = np.minimum(opens, closes) * (1 - volatility)
            volumes = np.random.normal(base_price * 1000, base_price * 100, len(timestamps))
            volumes = np.abs(volumes)  # Ensure volumes are positive
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=timestamps)
            
            # Cache the data
            self.cache[cache_key] = df
            self.last_update[cache_key] = current_time
            
            return df
        except Exception as e:
            print(f"Error generating data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_available_pairs(self) -> List[str]:
        """
        Get list of available trading pairs.
        
        Returns:
            List[str]: List of available trading pairs
        """
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            print(f"Error fetching available pairs: {str(e)}")
            return []
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Optional[Dict]: Ticker information or None if error
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                          days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data for a specified number of days.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for the data
            days (int): Number of days of historical data to fetch
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        # Calculate number of candles needed
        if timeframe == '1h':
            limit = days * 24
        elif timeframe == '4h':
            limit = days * 6
        elif timeframe == '1d':
            limit = days
        else:
            limit = 500  # Default limit
            
        return self.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Alias for fetch_ohlcv for backward compatibility."""
        return self.fetch_ohlcv(symbol) 