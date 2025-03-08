import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class TechDisruptor(BaseAgent):
    def __init__(self, name: str = "Elon Musk AI", timeframe: str = '1h'):
        """
        Initialize the Tech Disruptor agent inspired by Elon Musk's approach.
        Focuses on high-growth potential and innovative technologies.
        """
        super().__init__(
            name=name,
            personality="Tech Disruptor - Bold, innovative, high-growth focused",
            risk_tolerance=0.8,  # High risk tolerance
            timeframe=timeframe
        )
        
        # Set Elon Musk-inspired strategy preferences
        self.set_strategy_preferences({
            'innovation_focus': 0.9,
            'momentum_trading': 0.8,
            'trend_following': 0.7,
            'contrarian_plays': 0.6,
            'volatility_preference': 0.9
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'variable',
            'preferred_condition': 'high-growth',
            'market_approach': 'momentum-driven',
            'volatility_view': 'seeks-opportunity'
        })
        
        # Technical parameters
        self.momentum_window = 14
        self.volatility_window = 20
        self.volume_surge_threshold = 2.0
        self.trend_threshold = 0.05
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data using tech-focused momentum strategy.
        Looks for high momentum, volume surges, and breakout patterns.
        """
        df = market_data.copy()
        
        # Calculate momentum indicators
        df['RSI'] = ta.momentum.rsi(df['close'], window=self.momentum_window)
        df['MACD'] = ta.trend.macd_diff(df['close'])
        
        # Calculate volume metrics
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_surge'] = df['volume'] / df['volume_ma']
        
        # Calculate volatility and momentum metrics
        df['volatility'] = df['close'].pct_change().rolling(window=self.volatility_window).std()
        df['momentum'] = df['close'].pct_change(periods=self.momentum_window)
        
        # Get latest data
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate trend and momentum metrics
        price_momentum = latest['momentum']
        volume_momentum = latest['volume_surge']
        trend_strength = latest['MACD'] / df['MACD'].std()
        
        # Identify potential breakout patterns
        is_breakout = (
            latest['RSI'] > 70 and
            latest['volume_surge'] > self.volume_surge_threshold and
            price_momentum > self.trend_threshold
        )
        
        # Calculate innovation score (based on momentum and volume)
        innovation_score = (
            (latest['RSI'] / 100) * 0.3 +
            (min(latest['volume_surge'] / 3, 1)) * 0.3 +
            (min(abs(price_momentum) * 10, 1)) * 0.4
        )
        
        return {
            'price': latest['close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'volume_surge': latest['volume_surge'],
            'momentum': price_momentum,
            'trend_strength': trend_strength,
            'is_breakout': is_breakout,
            'innovation_score': innovation_score,
            'volatility': latest['volatility'],
            'timestamp': latest.name
        }
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on tech disruption principles.
        Focuses on high momentum and breakthrough opportunities.
        """
        signal = {
            'action': 'hold',
            'price': analysis['price'],
            'confidence': 0.0,
            'timestamp': analysis['timestamp']
        }
        
        # Base confidence on momentum and innovation metrics
        base_confidence = min(
            analysis['innovation_score'] * 0.6 +
            abs(analysis['trend_strength']) * 0.4,
            1.0
        )
        
        # Strong buy conditions (breakthrough opportunity)
        if (analysis['is_breakout'] and
            analysis['momentum'] > self.trend_threshold and
            analysis['volume_surge'] > self.volume_surge_threshold):
            
            signal['action'] = 'buy'
            signal['confidence'] = base_confidence * 0.95  # High confidence for strong momentum
            
        # Sell conditions (momentum reversal)
        elif (analysis['rsi'] < 30 and
              analysis['momentum'] < -self.trend_threshold and
              analysis['trend_strength'] < 0):
            
            signal['action'] = 'sell'
            signal['confidence'] = base_confidence * 0.85
        
        # Add Elon Musk-style commentary
        signal['commentary'] = self._generate_musk_commentary(analysis)
        
        return signal
    
    def _generate_musk_commentary(self, analysis: Dict) -> str:
        """Generate Elon Musk-style market commentary."""
        if analysis['is_breakout']:
            return "To the moon! 🚀"
        elif analysis['innovation_score'] > 0.8:
            return "The future is now! 💫"
        elif analysis['volume_surge'] > self.volume_surge_threshold:
            return "Funding secured! 💎"
        elif analysis['momentum'] < -self.trend_threshold:
            return "Time to disrupt the dip! 🔄"
        else:
            return "HODL! Diamond hands! 💎🙌" 