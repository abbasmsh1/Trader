"""
Binance API service for fetching market data.
"""

from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import requests

class BinanceService:
    def __init__(self):
        """Initialize Binance client."""
        # Initialize without API keys for public data only
        self.client = Client()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        self.base_url = "https://api.binance.com/api/v3"
        self.klines_url = f"{self.base_url}/klines"
        self.ticker_url = f"{self.base_url}/ticker/24hr"
        self.ticker_price_url = f"{self.base_url}/ticker/price"
        
    def get_market_prices(self) -> Dict[str, Any]:
        """Get current market prices and 24h changes."""
        try:
            market_data = {}
            for symbol in self.symbols:
                binance_symbol = symbol.replace('/', '')
                ticker = self.client.get_ticker(symbol=binance_symbol)
                market_data[symbol] = {
                    'price': float(ticker['lastPrice']),
                    'change_24h': float(ticker['priceChangePercent'])
                }
            return market_data
        except BinanceAPIException as e:
            print(f"Error fetching market data: {str(e)}")
            return {}
            
    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price and 24h stats for a symbol."""
        try:
            # Convert symbol format (e.g., 'BTC/USDT' to 'BTCUSDT')
            binance_symbol = symbol.replace('/', '')
            
            # Get 24h ticker stats
            ticker_response = requests.get(f"{self.ticker_url}?symbol={binance_symbol}")
            if not ticker_response.ok:
                return None
                
            ticker_data = ticker_response.json()
            
            # Get current price
            price_response = requests.get(f"{self.ticker_price_url}?symbol={binance_symbol}")
            if not price_response.ok:
                return None
                
            price_data = price_response.json()
            
            return {
                'price': float(price_data['price']),
                'change_24h': float(ticker_data['priceChangePercent']),
                'volume': float(ticker_data['volume']),
                'high_24h': float(ticker_data['highPrice']),
                'low_24h': float(ticker_data['lowPrice'])
            }
        except Exception as e:
            print(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def get_historical_prices(self, symbol: str, interval: str = '1h') -> List[Dict[str, Any]]:
        """Get historical price data for charting."""
        try:
            # Convert symbol format (e.g., 'BTC/USDT' to 'BTCUSDT')
            binance_symbol = symbol.replace('/', '')
            
            # Calculate start time (30 days ago)
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
            # Get klines data
            response = requests.get(
                f"{self.klines_url}?symbol={binance_symbol}&interval={interval}&startTime={start_time}&endTime={end_time}"
            )
            
            if not response.ok:
                return []
                
            klines_data = response.json()
            
            # Format data
            return [
                {
                    'timestamp': kline[0],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                }
                for kline in klines_data
            ]
        except Exception as e:
            print(f"Error getting historical prices for {symbol}: {str(e)}")
            return [] 