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
        Analyze market data using tech-focused metrics.
        Looks for momentum, breakouts, and innovation indicators.
        """
        df = market_data.copy()
        
        # Calculate technical indicators
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])
        df['MACD_diff'] = ta.trend.macd_diff(df['close'])
        df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])
        
        # Calculate volume surge
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_surge'] = df['volume'] / df['volume_ma']
        
        # Calculate momentum
        df['momentum'] = df['close'].pct_change(periods=5)
        
        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(window=10).std()
        
        # Get current values
        close_price = float(df['close'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        macd_signal = float(df['MACD_signal'].iloc[-1])
        macd_diff = float(df['MACD_diff'].iloc[-1])
        bb_upper = float(df['BB_upper'].iloc[-1])
        bb_lower = float(df['BB_lower'].iloc[-1])
        volume_surge = float(df['volume_surge'].iloc[-1])
        momentum = float(df['momentum'].iloc[-1])
        volatility = float(df['volatility'].iloc[-1])
        
        # Calculate trend strength (0 to 1)
        trend_strength = min(1.0, max(0.0, abs(momentum) * 10))
        
        # Determine if there's a breakout
        is_breakout = (close_price > bb_upper) or (volume_surge > 2.0 and momentum > 0.05)
        
        # Calculate innovation score (proxy for technological advancement)
        innovation_score = 0.5 + (0.5 * (
            (0.4 * (rsi / 100)) +  # RSI component
            (0.3 * (1 if macd > macd_signal else -1) * min(1.0, abs(macd_diff) * 5)) +  # MACD component
            (0.3 * min(1.0, volume_surge / 3))  # Volume component
        ))
        
        # Calculate overall confidence
        confidence = (
            innovation_score * 0.5 +  # Innovation component
            trend_strength * 0.3 +  # Trend component
            (1.0 if is_breakout else 0.0) * 0.2  # Breakout component
        )
        
        # Determine trend direction
        trend_direction = 1 if momentum > 0 else -1
        
        # Prepare analysis results
        analysis = {
            'current_price': close_price,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_diff': macd_diff,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'volume_surge': volume_surge,
            'momentum': momentum,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'is_breakout': is_breakout,
            'innovation_score': innovation_score,
            'confidence': confidence,
            'trend_direction': trend_direction
        }
        
        return analysis
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on tech disruption principles and news sentiment.
        Focuses on high momentum and breakthrough opportunities.
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['current_price']),
            'confidence': float(analysis['confidence']),
            'timestamp': analysis['timestamp']
        }
        
        # Base confidence on momentum and innovation metrics
        base_confidence = float(analysis['confidence'])
        
        # Adjust confidence based on news sentiment
        adjusted_confidence = self.adjust_confidence(base_confidence, analysis['sentiment'])
        
        # Explosive breakout opportunity with positive news
        if (analysis['is_breakout'] and
            float(analysis['momentum']) > self.trend_threshold * 2 and
            float(analysis['volume_surge']) > self.volume_surge_threshold * 1.5 and
            analysis['sentiment']['sentiment_score'] > 0.3):
            
            signal['action'] = 'STRONG_BUY'
            signal['confidence'] = adjusted_confidence * 0.95
            
        # Strong momentum entry
        elif (analysis['is_breakout'] and
              float(analysis['momentum']) > self.trend_threshold and
              float(analysis['volume_surge']) > self.volume_surge_threshold):
            
            signal['action'] = 'BUY'
            signal['confidence'] = adjusted_confidence * 0.85
            
        # Building momentum
        elif (float(analysis['momentum']) > self.trend_threshold * 0.5 and
              float(analysis['volume_surge']) > self.volume_surge_threshold * 0.8):
            
            signal['action'] = 'SCALE_IN'
            signal['confidence'] = adjusted_confidence * 0.75
            
        # Strong reversal with negative news
        elif (float(analysis['rsi']) < 20 and
              float(analysis['momentum']) < -self.trend_threshold * 2 and
              float(analysis['trend_strength']) < -0.2 or
              analysis['sentiment']['sentiment_score'] < -0.3):
            
            signal['action'] = 'STRONG_SELL'
            signal['confidence'] = adjusted_confidence * 0.9
            
        # Momentum breakdown
        elif (float(analysis['rsi']) < 30 and
              float(analysis['momentum']) < -self.trend_threshold and
              float(analysis['trend_strength']) < 0):
            
            signal['action'] = 'SELL'
            signal['confidence'] = adjusted_confidence * 0.8
            
        # Taking profits
        elif (float(analysis['rsi']) > 80 or
              float(analysis['momentum']) < -self.trend_threshold * 0.5):
            
            signal['action'] = 'SCALE_OUT'
            signal['confidence'] = adjusted_confidence * 0.7
            
        # Monitoring conditions
        elif (abs(float(analysis['momentum'])) < self.trend_threshold * 0.3 or
              float(analysis['volume_surge']) < 1.0):
            
            signal['action'] = 'WATCH'
            signal['confidence'] = adjusted_confidence * 0.6
        
        # Add Elon Musk-style commentary with news sentiment
        signal['commentary'] = self._generate_musk_commentary(analysis, signal['action'])
        
        return signal
    
    def _generate_musk_commentary(self, analysis: Dict, action: str) -> str:
        """Generate Elon Musk-style market commentary based on the action and news."""
        base_comment = ""
        if action == 'STRONG_BUY':
            base_comment = "🚀 To Mars! Diamond hands activated! This is the way! 💎🙌"
        elif action == 'BUY':
            base_comment = "Funding secured! Time to launch! 🚀💫"
        elif action == 'SCALE_IN':
            base_comment = "Building the future, one position at a time! 🏗️🔋"
        elif action == 'STRONG_SELL':
            base_comment = "Houston, we have a problem! Time to eject! 🔥"
        elif action == 'SELL':
            base_comment = "Not stonks! Moving capital to Mars colony! 📉🛸"
        elif action == 'SCALE_OUT':
            base_comment = "Taking some profits to fund the next moonshot! 🌙💰"
        elif action == 'WATCH':
            base_comment = "Doing due diligence... might delete later! 🤔"
        else:
            base_comment = "HODL! We're still early! 💎🙌"
        
        # Add news sentiment commentary with tech-focused spin
        news_comment = self.get_news_commentary(analysis['sentiment'])
        innovation_comment = f"Innovation Score: {analysis['innovation_score']:.2f} 🔬"
        
        return f"{base_comment}\n{innovation_comment}\n{news_comment}" 