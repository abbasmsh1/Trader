import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class TrendFollower(BaseAgent):
    def __init__(self, name: str = "Trend Follower", risk_tolerance: float = 0.6, timeframe: str = '4h'):
        """
        Initialize the Trend Follower agent.
        This agent follows market trends using moving averages and momentum indicators.
        """
        super().__init__(
            name=name,
            personality="Technical Trader - Systematic, trend-focused, data-driven",
            risk_tolerance=risk_tolerance,
            timeframe=timeframe
        )
        self.short_window = 20
        self.long_window = 50
        self.rsi_period = 14
        self.rsi_overbought = 70.0  # Ensure float
        self.rsi_oversold = 30.0    # Ensure float
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data using trend-following indicators.
        
        Args:
            market_data (pd.DataFrame): Historical market data with OHLCV
            
        Returns:
            Dict: Analysis results including trend direction and strength
        """
        df = market_data.copy()
        
        # Calculate moving averages
        df['SMA_short'] = ta.trend.sma_indicator(df['close'], self.short_window)
        df['SMA_long'] = ta.trend.sma_indicator(df['close'], self.long_window)
        
        # Calculate RSI
        df['RSI'] = ta.momentum.rsi(df['close'], self.rsi_period)
        
        # Calculate MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Ensure numeric values and handle NaN
        sma_short = float(latest['SMA_short']) if pd.notnull(latest['SMA_short']) else 0.0
        sma_long = float(latest['SMA_long']) if pd.notnull(latest['SMA_long']) else 0.0
        rsi = float(latest['RSI']) if pd.notnull(latest['RSI']) else 50.0
        macd_current = float(latest['MACD']) if pd.notnull(latest['MACD']) else 0.0
        macd_signal = float(latest['MACD_signal']) if pd.notnull(latest['MACD_signal']) else 0.0
        macd_prev = float(prev['MACD']) if pd.notnull(prev['MACD']) else 0.0
        macd_signal_prev = float(prev['MACD_signal']) if pd.notnull(prev['MACD_signal']) else 0.0
        
        # Determine trend direction and strength
        trend_direction = 1 if sma_short > sma_long else -1
        trend_strength = abs(sma_short - sma_long) / float(latest['close'])
        
        # Check for MACD crossover
        macd_crossover = (
            (macd_current > macd_signal and macd_prev <= macd_signal_prev) or
            (macd_current < macd_signal and macd_prev >= macd_signal_prev)
        )
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': float(trend_strength),
            'rsi': float(rsi),
            'macd_crossover': bool(macd_crossover),
            'price': float(latest['close']),
            'timestamp': latest.name
        }
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on trend analysis.
        
        Args:
            analysis (Dict): Market analysis results
            
        Returns:
            Dict: Trading signal with action, price, and confidence
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['price']),
            'confidence': 0.0,
            'timestamp': analysis['timestamp']
        }
        
        # Calculate base confidence from trend strength
        base_confidence = min(float(analysis['trend_strength']) * 10, 1.0)
        
        # Strong uptrend with confirmation
        if (analysis['trend_direction'] > 0 and
            float(analysis['rsi']) < self.rsi_overbought - 10 and
            analysis['macd_crossover'] and
            base_confidence > 0.8):
            
            signal['action'] = 'STRONG_BUY'
            signal['confidence'] = base_confidence * 0.95
            
        # Regular uptrend entry
        elif (analysis['trend_direction'] > 0 and
              float(analysis['rsi']) < self.rsi_overbought and
              analysis['macd_crossover']):
            
            signal['action'] = 'BUY'
            signal['confidence'] = base_confidence * 0.85
            
        # Building position in uptrend
        elif (analysis['trend_direction'] > 0 and
              float(analysis['rsi']) < self.rsi_overbought + 5):
            
            signal['action'] = 'SCALE_IN'
            signal['confidence'] = base_confidence * 0.75
            
        # Strong downtrend with confirmation
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold + 10 and
              analysis['macd_crossover'] and
              base_confidence > 0.8):
            
            signal['action'] = 'STRONG_SELL'
            signal['confidence'] = base_confidence * 0.9
            
        # Regular downtrend exit
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold and
              analysis['macd_crossover']):
            
            signal['action'] = 'SELL'
            signal['confidence'] = base_confidence * 0.8
            
        # Reducing position in downtrend
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold - 5):
            
            signal['action'] = 'SCALE_OUT'
            signal['confidence'] = base_confidence * 0.7
            
        # Monitoring conditions
        elif abs(analysis['trend_direction']) < 0.5:
            signal['action'] = 'WATCH'
            signal['confidence'] = base_confidence * 0.6
        
        # Add technical analysis commentary
        signal['commentary'] = self._generate_technical_commentary(analysis, signal['action'])
        
        return signal
    
    def _generate_technical_commentary(self, analysis: Dict, action: str) -> str:
        """Generate technical analysis commentary based on the action."""
        if action == 'STRONG_BUY':
            return "Strong bullish trend confirmed! Multiple indicators showing buy signals! 📈⬆️"
        elif action == 'BUY':
            return "Bullish trend developing with positive MACD crossover. 📈"
        elif action == 'SCALE_IN':
            return "Uptrend continues, adding to position on pullback. 📊"
        elif action == 'STRONG_SELL':
            return "Strong bearish trend confirmed! Multiple indicators showing sell signals! 📉⬇️"
        elif action == 'SELL':
            return "Bearish trend developing with negative MACD crossover. 📉"
        elif action == 'SCALE_OUT':
            return "Downtrend continues, reducing position on bounce. 📊"
        elif action == 'WATCH':
            return "Consolidation phase - waiting for clear trend direction. 🔍"
        else:
            return "Neutral trend - maintaining current position. ⚖️" 