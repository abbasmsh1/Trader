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
        self.medium_window = 50
        self.long_window = 100
        self.rsi_period = 14
        self.rsi_overbought = 70.0  # Ensure float
        self.rsi_oversold = 30.0    # Ensure float
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data using trend following principles.
        Looks for strong trends and momentum.
        """
        df = market_data.copy()
        
        # Calculate moving averages
        df['SMA_short'] = ta.trend.sma_indicator(df['close'], window=self.short_window)
        df['SMA_medium'] = ta.trend.sma_indicator(df['close'], window=self.medium_window)
        df['SMA_long'] = ta.trend.sma_indicator(df['close'], window=self.long_window)
        
        # Calculate momentum indicators
        df['RSI'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        df['MACD'] = ta.trend.macd_diff(df['close'])
        
        # Calculate ADX for trend strength
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # Get current values
        close_price = float(df['close'].iloc[-1])
        sma_short = float(df['SMA_short'].iloc[-1])
        sma_medium = float(df['SMA_medium'].iloc[-1])
        sma_long = float(df['SMA_long'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        adx = float(df['ADX'].iloc[-1])
        
        # Calculate trend metrics
        short_medium_trend = sma_short - sma_medium
        medium_long_trend = sma_medium - sma_long
        
        # Determine trend direction (1 for up, -1 for down, 0 for sideways)
        if short_medium_trend > 0 and medium_long_trend > 0:
            trend_direction = 1  # Strong uptrend
        elif short_medium_trend < 0 and medium_long_trend < 0:
            trend_direction = -1  # Strong downtrend
        elif short_medium_trend > 0 and medium_long_trend < 0:
            trend_direction = 0.5  # Potential trend reversal (up)
        elif short_medium_trend < 0 and medium_long_trend > 0:
            trend_direction = -0.5  # Potential trend reversal (down)
        else:
            trend_direction = 0  # No clear trend
        
        # Calculate trend strength (0 to 1)
        trend_strength = min(1.0, adx / 50.0)  # ADX above 25 indicates strong trend
        
        # Calculate momentum alignment with trend
        momentum_alignment = 1.0 if (trend_direction > 0 and rsi > 50) or (trend_direction < 0 and rsi < 50) else 0.5
        
        # Calculate overall confidence
        confidence = (
            trend_strength * 0.6 +  # Trend strength component
            momentum_alignment * 0.4  # Momentum alignment component
        )
        
        # Prepare analysis results
        analysis = {
            'current_price': close_price,
            'sma_short': sma_short,
            'sma_medium': sma_medium,
            'sma_long': sma_long,
            'rsi': rsi,
            'macd': macd,
            'adx': adx,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'momentum_alignment': momentum_alignment,
            'confidence': confidence
        }
        
        return analysis
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on trend analysis and news sentiment.
        
        Args:
            analysis (Dict): Market analysis results
            
        Returns:
            Dict: Trading signal with action, price, and confidence
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['current_price']),
            'confidence': float(analysis['confidence']),
            'timestamp': analysis['timestamp']
        }
        
        # Strong uptrend with confirmation and positive news
        if (analysis['trend_direction'] > 0 and
            float(analysis['rsi']) < self.rsi_overbought - 10 and
            analysis['momentum_alignment'] > 0.8 and
            analysis['confidence'] > 0.8):
            
            signal['action'] = 'STRONG_BUY'
            signal['confidence'] = analysis['confidence'] * 0.95
            
        # Regular uptrend entry
        elif (analysis['trend_direction'] > 0 and
              float(analysis['rsi']) < self.rsi_overbought and
              analysis['momentum_alignment'] > 0.5):
            
            signal['action'] = 'BUY'
            signal['confidence'] = analysis['confidence'] * 0.85
            
        # Building position in uptrend
        elif (analysis['trend_direction'] > 0 and
              float(analysis['rsi']) < self.rsi_overbought + 5):
            
            signal['action'] = 'SCALE_IN'
            signal['confidence'] = analysis['confidence'] * 0.75
            
        # Strong downtrend with confirmation and negative news
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold + 10 and
              analysis['momentum_alignment'] < 0.2 and
              analysis['confidence'] > 0.8):
            
            signal['action'] = 'STRONG_SELL'
            signal['confidence'] = analysis['confidence'] * 0.9
            
        # Regular downtrend exit
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold and
              analysis['momentum_alignment'] < 0.5):
            
            signal['action'] = 'SELL'
            signal['confidence'] = analysis['confidence'] * 0.8
            
        # Reducing position in downtrend
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold - 5):
            
            signal['action'] = 'SCALE_OUT'
            signal['confidence'] = analysis['confidence'] * 0.7
            
        # Monitoring conditions
        elif abs(analysis['trend_direction']) < 0.5:
            signal['action'] = 'WATCH'
            signal['confidence'] = analysis['confidence'] * 0.6
        
        # Add technical analysis commentary with news sentiment
        signal['commentary'] = self._generate_technical_commentary(analysis, signal['action'])
        
        return signal
    
    def _generate_technical_commentary(self, analysis: Dict, action: str) -> str:
        """Generate technical analysis commentary based on the action and news."""
        base_comment = ""
        if action == 'STRONG_BUY':
            base_comment = "Strong bullish trend confirmed! Multiple indicators showing buy signals! 📈⬆️"
        elif action == 'BUY':
            base_comment = "Bullish trend developing with positive MACD crossover. 📈"
        elif action == 'SCALE_IN':
            base_comment = "Uptrend continues, adding to position on pullback. 📊"
        elif action == 'STRONG_SELL':
            base_comment = "Strong bearish trend confirmed! Multiple indicators showing sell signals! 📉⬇️"
        elif action == 'SELL':
            base_comment = "Bearish trend developing with negative MACD crossover. 📉"
        elif action == 'SCALE_OUT':
            base_comment = "Downtrend continues, reducing position on bounce. 📊"
        elif action == 'WATCH':
            base_comment = "Consolidation phase - waiting for clear trend direction. 🔍"
        else:
            base_comment = "Neutral trend - maintaining current position. ⚖️"
        
        # Add technical metrics
        tech_comment = (
            f"RSI: {analysis['rsi']:.1f} | " +
            f"Trend Strength: {analysis['trend_strength']:.2f} | " +
            f"Direction: {'Bullish' if analysis['trend_direction'] > 0 else 'Bearish'}"
        )
        
        # Add news sentiment commentary
        news_comment = self.get_news_commentary(analysis['sentiment'])
        
        return f"{base_comment}\n{tech_comment}\n{news_comment}" 