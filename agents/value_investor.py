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
            personality="Value Investor - Patient, long-term focused, seeks intrinsic value, avoids speculation",
            risk_tolerance=0.3,  # Conservative risk profile
            timeframe=timeframe
        )
        
        # Warren Buffett's famous quotes
        self.quotes = [
            "Be fearful when others are greedy, and greedy when others are fearful.",
            "Price is what you pay. Value is what you get.",
            "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price.",
            "Our favorite holding period is forever.",
            "Risk comes from not knowing what you're doing.",
            "Only buy something that you'd be perfectly happy to hold if the market shut down for 10 years."
        ]
        
        # Set Warren Buffett-inspired strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.9,            # Strong emphasis on intrinsic value
            'trend_following': 0.2,        # Limited interest in trends
            'momentum': 0.1,               # Very low interest in momentum
            'mean_reversion': 0.7,         # Believes in reversion to fair value
            'volatility_preference': 0.2,  # Prefers stability over volatility
            'speculation': 0.0,            # Avoids pure speculation
            'meme_coin_interest': 0.0,     # No interest in meme coins
            'alt_coin_interest': 0.3       # Limited interest in established alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'very-long-term',
            'preferred_condition': 'undervalued',
            'market_approach': 'fundamental-driven',
            'volatility_view': 'prefers-stability',
            'crypto_view': 'skeptical-but-open',
            'meme_coin_view': 'avoid-completely',
            'innovation_view': 'cautious-adoption'
        })
        
        # Technical parameters
        self.ma_long = 200  # 200-day moving average for long-term trend
        self.ma_medium = 50  # 50-day moving average for medium-term trend
        self.volume_threshold = 1.5  # Volume increase threshold
        
    def get_personality_traits(self) -> Dict[str, str]:
        """Get personality traits of the agent."""
        traits = super().get_personality_traits()
        
        # Add Buffett-specific traits
        traits['personality'] = "Value"
        traits['investment_philosophy'] = "Buy and hold quality assets at reasonable prices"
        traits['famous_quote'] = np.random.choice(self.quotes)
        traits['risk_approach'] = "Conservative, focuses on capital preservation"
        traits['time_horizon'] = "Very long-term, prefers to hold indefinitely"
        traits['meme_coin_approach'] = "Avoids speculative assets without intrinsic value"
        
        return traits
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data using value investing principles and news sentiment.
        Looks for stable trends, consistent volume, and potential undervaluation.
        """
        df = market_data.copy()
        
        # Calculate moving averages for trend analysis
        df['MA_long'] = ta.trend.sma_indicator(df['close'], window=self.ma_long)
        df['MA_medium'] = ta.trend.sma_indicator(df['close'], window=self.ma_medium)
        
        # Calculate volume metrics
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Get latest data
        latest = df.iloc[-1]
        
        # Ensure numeric values and handle NaN
        close_price = float(latest['close']) if pd.notnull(latest['close']) else 0.0
        ma_long = float(latest['MA_long']) if pd.notnull(latest['MA_long']) else close_price
        ma_medium = float(latest['MA_medium']) if pd.notnull(latest['MA_medium']) else close_price
        volume_ratio = float(latest['volume_ratio']) if pd.notnull(latest['volume_ratio']) else 1.0
        volatility = float(latest['volatility']) if pd.notnull(latest['volatility']) else 0.0
        volume_ma = float(latest['volume_ma']) if pd.notnull(latest['volume_ma']) else 1.0
        
        # Calculate value metrics
        price_to_volume = close_price / volume_ma
        trend_strength = (ma_medium - ma_long) / ma_long
        
        # Determine if the asset is potentially undervalued
        is_undervalued = (
            close_price < ma_long and
            volume_ratio > self.volume_threshold and
            volatility < df['volatility'].mean()
        )
        
        # Calculate market stability
        stability_score = 1 - (volatility / df['volatility'].max() if df['volatility'].max() > 0 else 0)
        
        # Get news sentiment analysis
        symbol = market_data.name if hasattr(market_data, 'name') else 'UNKNOWN'
        sentiment = self.analyze_news_sentiment(symbol)
        
        return {
            'price': close_price,
            'ma_long': ma_long,
            'ma_medium': ma_medium,
            'volume_ratio': volume_ratio,
            'trend_strength': float(trend_strength),
            'is_undervalued': bool(is_undervalued),
            'stability_score': float(stability_score),
            'volatility': volatility,
            'price_to_volume': float(price_to_volume),
            'sentiment': sentiment,
            'timestamp': latest.name
        }
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on value investing principles and news sentiment.
        Focuses on stable, undervalued assets with strong fundamentals.
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['price']),
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
        elif (float(analysis['price']) > float(analysis['ma_long']) * 1.8 or
              float(analysis['volatility']) > 0.15 or
              analysis['sentiment']['sentiment_score'] < -0.3):
            
            signal['action'] = 'STRONG_SELL'
            signal['confidence'] = adjusted_confidence * 0.9
            
        # Regular sell conditions
        elif (float(analysis['price']) > float(analysis['ma_long']) * 1.5 or
              float(analysis['volatility']) > 0.12):
            
            signal['action'] = 'SELL'
            signal['confidence'] = adjusted_confidence * 0.8
            
        # Gradual position reduction
        elif (float(analysis['price']) > float(analysis['ma_long']) * 1.3 or
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