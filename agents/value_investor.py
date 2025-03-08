import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class ValueInvestor(BaseAgent):
    def __init__(self, name: str = "Warren Buffett AI", timeframe: str = '1d'):
        """
        Initialize the Value Investor agent inspired by Warren Buffett's principles.
        Uses fundamental analysis and looks for undervalued assets with strong potential.
        """
        super().__init__(
            name=name,
            personality="Value Investor - Patient, long-term focused, seeks intrinsic value",
            risk_tolerance=0.3,  # Conservative risk profile
            timeframe=timeframe
        )
        
        # Set Warren Buffett-inspired strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.8,
            'trend_following': 0.2,
            'momentum': 0.1,
            'mean_reversion': 0.7,
            'volatility_preference': 0.3
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'long-term',
            'preferred_condition': 'undervalued',
            'market_approach': 'fundamental-driven',
            'volatility_view': 'prefers-stability'
        })
        
        # Technical parameters
        self.ma_long = 200  # 200-day moving average for long-term trend
        self.ma_medium = 50  # 50-day moving average for medium-term trend
        self.volume_threshold = 1.5  # Volume increase threshold
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data using value investing principles.
        Looks for stable trends, consistent volume, and potential undervaluation.
        """
        df = market_data.copy()
        
        # Calculate moving averages for trend analysis
        df['MA_long'] = ta.trend.sma_indicator(df['close'], window=self.ma_long)
        df['MA_medium'] = ta.trend.sma_indicator(df['close'], window=self.ma_medium)
        
        # Calculate volume indicators
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Calculate RSI for momentum
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Get current values
        current_close = float(df['close'].iloc[-1])
        current_ma_long = float(df['MA_long'].iloc[-1])
        current_ma_medium = float(df['MA_medium'].iloc[-1])
        current_volume_ratio = float(df['volume_ratio'].iloc[-1])
        current_volatility = float(df['volatility'].iloc[-1])
        current_rsi = float(df['rsi'].iloc[-1])
        
        # Calculate trend strength (0 to 1)
        trend_strength = min(1.0, max(0.0, (current_close / current_ma_long - 0.9) * 5))
        
        # Determine if the asset is undervalued
        # For crypto, we use a combination of technical indicators as a proxy
        is_undervalued = current_close < current_ma_long * 0.95 and current_rsi < 40
        
        # Calculate stability score (higher is more stable)
        stability_score = 1.0 - min(1.0, current_volatility * 10)
        
        # Calculate value score
        value_score = 0.6 if is_undervalued else 0.3
        
        # Adjust value score based on trend and stability
        value_score = value_score * (0.7 + 0.3 * stability_score)
        
        # Calculate overall confidence
        confidence = (
            value_score * 0.6 +  # Value component
            trend_strength * 0.3 +  # Trend component
            stability_score * 0.1  # Stability component
        )
        
        # Determine trend direction
        trend_direction = 1 if current_close > current_ma_medium else -1
        
        # Prepare analysis results
        analysis = {
            'current_price': current_close,
            'ma_long': current_ma_long,
            'ma_medium': current_ma_medium,
            'volume_ratio': current_volume_ratio,
            'volatility': current_volatility,
            'rsi': current_rsi,
            'trend_strength': trend_strength,
            'is_undervalued': is_undervalued,
            'stability_score': stability_score,
            'value_score': value_score,
            'confidence': confidence,
            'trend_direction': trend_direction
        }
        
        return analysis
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on value investing principles and news sentiment.
        Focuses on stable, undervalued assets with strong fundamentals.
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['current_price']),
            'confidence': 0.0,
            'timestamp': analysis['timestamp']
        }
        
        # Base confidence on stability and value metrics
        base_confidence = min(float(analysis['stability_score']) * 0.7 + 
                            (1 - float(analysis['volatility'])) * 0.3, 1.0)
        
        # Adjust confidence based on news sentiment
        adjusted_confidence = self.adjust_confidence(base_confidence, analysis['sentiment'])
        
        # Strong value opportunity conditions
        if (analysis['is_undervalued'] and
            float(analysis['volume_ratio']) > self.volume_threshold * 1.5 and
            float(analysis['stability_score']) > 0.8 and
            analysis['sentiment']['sentiment_score'] > 0.2):
            
            signal['action'] = 'STRONG_BUY'
            signal['confidence'] = adjusted_confidence * 0.95
            
        # Good value entry point
        elif (analysis['is_undervalued'] and
              float(analysis['volume_ratio']) > self.volume_threshold and
              float(analysis['stability_score']) > 0.7):
            
            signal['action'] = 'BUY'
            signal['confidence'] = adjusted_confidence * 0.85
            
        # Gradual position building
        elif (analysis['is_undervalued'] and
              float(analysis['stability_score']) > 0.6):
            
            signal['action'] = 'SCALE_IN'
            signal['confidence'] = adjusted_confidence * 0.75
            
        # Strong sell conditions
        elif (float(analysis['current_price']) > float(analysis['ma_long']) * 1.8 or
              float(analysis['volatility']) > 0.15 or
              analysis['sentiment']['sentiment_score'] < -0.3):
            
            signal['action'] = 'STRONG_SELL'
            signal['confidence'] = adjusted_confidence * 0.9
            
        # Regular sell conditions
        elif (float(analysis['current_price']) > float(analysis['ma_long']) * 1.5 or
              float(analysis['volatility']) > 0.12):
            
            signal['action'] = 'SELL'
            signal['confidence'] = adjusted_confidence * 0.8
            
        # Gradual position reduction
        elif (float(analysis['current_price']) > float(analysis['ma_long']) * 1.3 or
              float(analysis['volatility']) > 0.1):
            
            signal['action'] = 'SCALE_OUT'
            signal['confidence'] = adjusted_confidence * 0.7
            
        # Watch conditions
        elif (float(analysis['stability_score']) < 0.5 or
              abs(float(analysis['trend_strength'])) > 0.1):
            
            signal['action'] = 'WATCH'
            signal['confidence'] = adjusted_confidence * 0.6
        
        # Add Warren Buffett-style commentary with news sentiment
        signal['commentary'] = self._generate_buffett_commentary(analysis, signal['action'])
        
        return signal
    
    def _generate_buffett_commentary(self, analysis: Dict, action: str) -> str:
        """Generate Warren Buffett-style market commentary based on the action and news."""
        base_comment = ""
        if action == 'STRONG_BUY':
            base_comment = "Be fearful when others are greedy, and greedy when others are fearful. This is a prime opportunity! 💎"
        elif action == 'BUY':
            base_comment = "Price is what you pay, value is what you get. The market is offering a fair deal. 📈"
        elif action == 'SCALE_IN':
            base_comment = "Our favorite holding period is forever. Time to start building a position. 🏗️"
        elif action == 'STRONG_SELL':
            base_comment = "When we own portions of outstanding businesses with outstanding managements, our favorite holding period is forever. But not this time. ⚠️"
        elif action == 'SELL':
            base_comment = "The most important thing to do if you find yourself in a hole is to stop digging. 🛑"
        elif action == 'SCALE_OUT':
            base_comment = "Should you find yourself in a chronically leaking boat, energy devoted to changing vessels is likely to be more productive than energy devoted to patching leaks. 🚣"
        elif action == 'WATCH':
            base_comment = "Risk comes from not knowing what you're doing. Let's wait and observe. 👀"
        else:
            base_comment = "The stock market is designed to transfer money from the active to the patient. HODL! 💪"
        
        # Add news sentiment commentary
        news_comment = self.get_news_commentary(analysis['sentiment'])
        
        return f"{base_comment}\n{news_comment}" 