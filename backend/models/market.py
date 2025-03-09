import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import time

from ..api.binance_api import BinanceAPI
from ..config import TRADING_PAIRS, TIMEFRAMES, TECHNICAL_INDICATORS

logger = logging.getLogger(__name__)


class Market:
    """
    Market model for handling market data and technical indicators.
    """
    
    def __init__(self, binance_api: BinanceAPI):
        """
        Initialize the market model.
        
        Args:
            binance_api: The Binance API client
        """
        self.binance_api = binance_api
        self.trading_pairs = TRADING_PAIRS
        self.timeframes = TIMEFRAMES
        self.indicators_config = TECHNICAL_INDICATORS
        self.data_cache = {}
        self.last_update = {}
        self.update_interval = 60  # Update interval in seconds
    
    def get_current_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current market data for all trading pairs.
        
        Returns:
            A dictionary mapping trading pair symbols to their market data
        """
        current_time = time.time()
        result = {}
        
        for symbol in self.trading_pairs:
            # Check if we need to update the data
            if (symbol not in self.last_update or 
                current_time - self.last_update.get(symbol, 0) > self.update_interval):
                
                # Get market data
                market_data = self.binance_api.get_market_data(symbol)
                
                # Add technical indicators
                for timeframe in self.timeframes.values():
                    indicators = self._calculate_indicators(symbol, timeframe)
                    if indicators:
                        if 'indicators' not in market_data:
                            market_data['indicators'] = {}
                        market_data['indicators'][timeframe] = indicators
                
                # Cache the data
                self.data_cache[symbol] = market_data
                self.last_update[symbol] = current_time
            
            result[symbol] = self.data_cache[symbol]
        
        return result
    
    def get_pair_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market data for a specific trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            A dictionary containing the market data, or None if the symbol is not found
        """
        if symbol not in self.trading_pairs:
            return None
        
        current_time = time.time()
        
        # Check if we need to update the data
        if (symbol not in self.last_update or 
            current_time - self.last_update.get(symbol, 0) > self.update_interval):
            
            # Get market data
            market_data = self.binance_api.get_market_data(symbol)
            
            # Add technical indicators
            for timeframe in self.timeframes.values():
                indicators = self._calculate_indicators(symbol, timeframe)
                if indicators:
                    if 'indicators' not in market_data:
                        market_data['indicators'] = {}
                    market_data['indicators'][timeframe] = indicators
            
            # Cache the data
            self.data_cache[symbol] = market_data
            self.last_update[symbol] = current_time
        
        return self.data_cache.get(symbol)
    
    def _calculate_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for a trading pair.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            timeframe: The time interval (e.g., '1m', '5m', '1h')
            
        Returns:
            A dictionary containing the technical indicators
        """
        try:
            # Get klines data
            df = self.binance_api.get_klines(symbol, timeframe)
            if df.empty:
                return {}
            
            result = {}
            
            # Calculate RSI
            if 'RSI' in self.indicators_config:
                window = self.indicators_config['RSI']['window']
                result['RSI'] = self._calculate_rsi(df, window)
            
            # Calculate MACD
            if 'MACD' in self.indicators_config:
                fast = self.indicators_config['MACD']['fast']
                slow = self.indicators_config['MACD']['slow']
                signal = self.indicators_config['MACD']['signal']
                macd_data = self._calculate_macd(df, fast, slow, signal)
                result['MACD'] = macd_data
            
            # Calculate Bollinger Bands
            if 'BOLLINGER' in self.indicators_config:
                window = self.indicators_config['BOLLINGER']['window']
                std_dev = self.indicators_config['BOLLINGER']['std_dev']
                bollinger_data = self._calculate_bollinger_bands(df, window, std_dev)
                result['BOLLINGER'] = bollinger_data
            
            return result
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol} ({timeframe}): {e}")
            return {}
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int) -> Dict[str, Any]:
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            df: DataFrame containing price data
            window: The RSI window period
            
        Returns:
            A dictionary containing the RSI value and interpretation
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Interpret RSI
        if current_rsi < 30:
            interpretation = "oversold"
        elif current_rsi > 70:
            interpretation = "overbought"
        else:
            interpretation = "neutral"
        
        return {
            'value': current_rsi,
            'interpretation': interpretation,
            'history': rsi.dropna().tolist()[-10:]  # Last 10 values
        }
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> Dict[str, Any]:
        """
        Calculate the Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame containing price data
            fast: The fast EMA period
            slow: The slow EMA period
            signal: The signal line period
            
        Returns:
            A dictionary containing the MACD values and interpretation
        """
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Interpret MACD
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            interpretation = "bullish_crossover"
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            interpretation = "bearish_crossover"
        elif macd_line.iloc[-1] > signal_line.iloc[-1]:
            interpretation = "bullish"
        else:
            interpretation = "bearish"
        
        return {
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'interpretation': interpretation,
            'history': {
                'macd': macd_line.dropna().tolist()[-10:],
                'signal': signal_line.dropna().tolist()[-10:],
                'histogram': histogram.dropna().tolist()[-10:]
            }
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int, std_dev: int) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame containing price data
            window: The window period
            std_dev: The number of standard deviations
            
        Returns:
            A dictionary containing the Bollinger Bands values and interpretation
        """
        # Calculate middle band (SMA)
        middle_band = df['close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        std = df['close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        current_price = df['close'].iloc[-1]
        current_middle = middle_band.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Calculate bandwidth and %B
        bandwidth = (current_upper - current_lower) / current_middle
        percent_b = (current_price - current_lower) / (current_upper - current_lower)
        
        # Interpret Bollinger Bands
        if current_price > current_upper:
            interpretation = "overbought"
        elif current_price < current_lower:
            interpretation = "oversold"
        elif current_price > current_middle and percent_b > 0.8:
            interpretation = "approaching_upper"
        elif current_price < current_middle and percent_b < 0.2:
            interpretation = "approaching_lower"
        else:
            interpretation = "neutral"
        
        return {
            'upper': current_upper,
            'middle': current_middle,
            'lower': current_lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b,
            'interpretation': interpretation,
            'history': {
                'upper': upper_band.dropna().tolist()[-10:],
                'middle': middle_band.dropna().tolist()[-10:],
                'lower': lower_band.dropna().tolist()[-10:]
            }
        } 