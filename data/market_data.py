import ccxt
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

class MarketDataFetcher:
    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize the market data fetcher.
        
        Args:
            exchange_id (str): ID of the exchange to use (default: 'binance')
        """
        load_dotenv()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize exchange
        try:
            print(f"Initializing {exchange_id} exchange...")
            exchange_class = getattr(ccxt, exchange_id)
            api_key = os.getenv(f'{exchange_id.upper()}_API_KEY')
            api_secret = os.getenv(f'{exchange_id.upper()}_API_SECRET')
            
            if not api_key or not api_secret:
                print(f"Warning: Missing API credentials for {exchange_id}")
            
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
            
            # Test connection
            self.exchange.load_markets()
            print(f"Successfully connected to {exchange_id}")
            
        except Exception as e:
            print(f"Error initializing exchange {exchange_id}: {str(e)}")
            raise
    
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
            print(f"Fetching OHLCV data for {symbol} ({timeframe} timeframe, {limit} candles)...")
            
            # Check if the symbol is available
            if symbol not in self.exchange.markets:
                print(f"Error: Symbol {symbol} not available on {self.exchange.id}")
                return pd.DataFrame()
            
            # Fetch the OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                print(f"Error: No OHLCV data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except ccxt.NetworkError as e:
            print(f"Network error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            print(f"Exchange error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Unexpected error fetching data for {symbol}: {str(e)}")
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
        try:
            print(f"Getting historical data for {symbol}...")
            
            # Calculate number of candles needed
            if timeframe == '1h':
                limit = days * 24
            elif timeframe == '4h':
                limit = days * 6
            elif timeframe == '1d':
                limit = days
            else:
                limit = 500  # Default limit
            
            print(f"Requesting {limit} candles for {symbol}")
            df = self.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if df.empty:
                print(f"Warning: No historical data available for {symbol}")
            else:
                print(f"Successfully retrieved {len(df)} historical candles for {symbol}")
            
            return df
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame() 