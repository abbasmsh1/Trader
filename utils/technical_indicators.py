"""
Technical indicators utilities for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        prices: Series of prices
        period: RSI period
        
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices: pd.Series, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Series of prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    # Calculate EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(prices: pd.Series, 
                            period: int = 20, 
                            std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of prices
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = prices.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_atr(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: ATR period
        
    Returns:
        Series of ATR values
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_stochastic(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series, 
                        k_period: int = 14, 
                        d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        k_period: %K period
        d_period: %D period
        
    Returns:
        Tuple of (%K line, %D line)
    """
    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D
    d_line = k_line.rolling(window=d_period).mean()
    
    return k_line, d_line

def calculate_volume_profile(prices: pd.Series, 
                           volume: pd.Series, 
                           num_bins: int = 20) -> Tuple[List[float], List[float]]:
    """
    Calculate Volume Profile.
    
    Args:
        prices: Series of prices
        volume: Series of volumes
        num_bins: Number of price bins
        
    Returns:
        Tuple of (price levels, volume at each level)
    """
    # Create price bins
    min_price = prices.min()
    max_price = prices.max()
    bin_size = (max_price - min_price) / num_bins
    bins = [min_price + i * bin_size for i in range(num_bins + 1)]
    
    # Calculate volume in each bin
    volume_profile = []
    for i in range(num_bins):
        mask = (prices >= bins[i]) & (prices < bins[i + 1])
        volume_profile.append(volume[mask].sum())
    
    # Use bin midpoints as price levels
    price_levels = [(bins[i] + bins[i + 1]) / 2 for i in range(num_bins)]
    
    return price_levels, volume_profile

def calculate_ichimoku_cloud(high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series,
                           conversion_period: int = 9,
                           base_period: int = 26,
                           leading_span_b_period: int = 52,
                           displacement: int = 26) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Ichimoku Cloud.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        conversion_period: Conversion line period
        base_period: Base line period
        leading_span_b_period: Leading span B period
        displacement: Cloud displacement
        
    Returns:
        Tuple of (conversion line, base line, leading span A, leading span B, lagging span)
    """
    # Calculate conversion line
    conversion_line = (high.rolling(window=conversion_period).max() + 
                      low.rolling(window=conversion_period).min()) / 2
    
    # Calculate base line
    base_line = (high.rolling(window=base_period).max() + 
                low.rolling(window=base_period).min()) / 2
    
    # Calculate leading span A
    leading_span_a = (conversion_line + base_line) / 2
    
    # Calculate leading span B
    leading_span_b = (high.rolling(window=leading_span_b_period).max() + 
                     low.rolling(window=leading_span_b_period).min()) / 2
    
    # Calculate lagging span
    lagging_span = close.shift(-displacement)
    
    return conversion_line, base_line, leading_span_a, leading_span_b, lagging_span 