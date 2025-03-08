import ccxt
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

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
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for the data (e.g., '1h', '4h', '1d')
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            # Fetch the OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
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