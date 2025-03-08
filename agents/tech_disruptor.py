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
        Analyze market data using tech-focused momentum strategy and news sentiment.
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
        
        # Ensure numeric values and handle NaN
        rsi = float(latest['RSI']) if pd.notnull(latest['RSI']) else 50.0
        macd = float(latest['MACD']) if pd.notnull(latest['MACD']) else 0.0
        volume_surge = float(latest['volume_surge']) if pd.notnull(latest['volume_surge']) else 1.0
        momentum = float(latest['momentum']) if pd.notnull(latest['momentum']) else 0.0
        volatility = float(latest['volatility']) if pd.notnull(latest['volatility']) else 0.0
        close_price = float(latest['close']) if pd.notnull(latest['close']) else 0.0
        
        # Calculate trend and momentum metrics
        price_momentum = momentum
        volume_momentum = volume_surge
        trend_strength = macd / (df['MACD'].std() if df['MACD'].std() > 0 else 1.0)
        
        # Identify potential breakout patterns
        is_breakout = (
            rsi > 70 and
            volume_surge > self.volume_surge_threshold and
            price_momentum > self.trend_threshold
        )
        
        # Calculate innovation score (based on momentum and volume)
        innovation_score = (
            (rsi / 100) * 0.3 +
            (min(volume_surge / 3, 1)) * 0.3 +
            (min(abs(price_momentum) * 10, 1)) * 0.4
        )
        
        # Get news sentiment analysis
        symbol = market_data.name if hasattr(market_data, 'name') else 'UNKNOWN'
        sentiment = self.analyze_news_sentiment(symbol)
        
        # Adjust innovation score based on news sentiment
        if sentiment['article_count'] > 0:
            innovation_score = innovation_score * (1 + sentiment['sentiment_score'] * 0.2)
        
        return {
            'price': close_price,
            'rsi': rsi,
            'macd': macd,
            'volume_surge': volume_surge,
            'momentum': float(price_momentum),
            'trend_strength': float(trend_strength),
            'is_breakout': bool(is_breakout),
            'innovation_score': float(innovation_score),
            'volatility': volatility,
            'sentiment': sentiment,
            'timestamp': latest.name
        }
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on tech disruption principles and news sentiment.
        Focuses on high momentum and breakthrough opportunities.
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['price']),
            'confidence': 0.0,
            'timestamp': analysis['timestamp']
        }
        
        # Base confidence on momentum and innovation metrics
        base_confidence = min(
            float(analysis['innovation_score']) * 0.6 +
            abs(float(analysis['trend_strength'])) * 0.4,
            1.0
        )
        
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