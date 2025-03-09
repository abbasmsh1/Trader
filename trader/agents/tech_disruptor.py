import pandas as pd
import numpy as np
from typing import Dict
import ta

from trader.agents.base_agent import BaseAgent

class TechDisruptor(BaseAgent):
    """
    Tech Disruptor agent inspired by Elon Musk's approach.
    Focuses on innovation, disruption, and high-growth potential.
    """
    def __init__(self, name: str = "Elon Musk AI", timeframe: str = '1h'):
        """
        Initialize the Tech Disruptor agent.
        
        Args:
            name: Name of the agent
            timeframe: Trading timeframe
        """
        super().__init__(
            name=name,
            personality="Tech Disruptor - Innovation-focused, high-risk tolerance, tech-oriented, embraces volatility",
            risk_tolerance=0.8,  # High risk tolerance
            timeframe=timeframe
        )
        
        # Elon Musk's style quotes
        self.quotes = [
            "The first step is to establish that something is possible; then probability will occur.",
            "When something is important enough, you do it even if the odds are not in your favor.",
            "I think it is possible for ordinary people to choose to be extraordinary.",
            "Persistence is very important. You should not give up unless you are forced to give up.",
            "I'd rather be optimistic and wrong than pessimistic and right.",
            "Some people don't like change, but you need to embrace change if the alternative is disaster."
        ]
        
        # Set Elon Musk-inspired strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.2,            # Limited interest in traditional value
            'trend_following': 0.7,        # Strong interest in trends
            'momentum': 0.9,               # Very high interest in momentum
            'mean_reversion': 0.1,         # Limited interest in mean reversion
            'volatility_preference': 0.8,  # Embraces volatility
            'speculation': 0.7,            # Willing to speculate
            'meme_coin_interest': 0.9,     # High interest in meme coins
            'alt_coin_interest': 0.8       # High interest in alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'medium-to-long-term',
            'preferred_condition': 'high-growth-potential',
            'market_approach': 'innovation-driven',
            'volatility_view': 'embraces-volatility',
            'crypto_view': 'enthusiastic-believer',
            'meme_coin_view': 'potential-disruptors',
            'innovation_view': 'early-adopter'
        })
        
    def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Trading pair symbol
            data: DataFrame with market data
            
        Returns:
            Dictionary with analysis results
        """
        if data.empty or len(data) < 20:
            return {'confidence': 0, 'reason': "Insufficient data for analysis"}
            
        # Tech disruptors focus on momentum, innovation, and growth potential
        
        # 1. Check for strong momentum
        current_price = data['close'].iloc[-1]
        prev_price_1d = data['close'].iloc[-2] if len(data) >= 2 else current_price
        prev_price_7d = data['close'].iloc[-7] if len(data) >= 7 else current_price
        
        momentum_1d = (current_price - prev_price_1d) / prev_price_1d
        momentum_7d = (current_price - prev_price_7d) / prev_price_7d
        
        # 2. Check for increasing volume (market interest)
        recent_volume = data['volume'].iloc[-3:].mean()
        earlier_volume = data['volume'].iloc[-10:-3].mean()
        volume_change = (recent_volume - earlier_volume) / earlier_volume if earlier_volume > 0 else 0
        
        # 3. Check for breakouts (tech disruptors love breakouts)
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else data['close'].mean()
        price_to_sma = (current_price - sma_20) / sma_20
        
        # 4. Check for volatility (tech disruptors embrace volatility)
        volatility = data['close'].pct_change().std()
        
        # Calculate RSI for momentum confirmation
        rsi = ta.momentum.RSIIndicator(data['close'], window=14).rsi().iloc[-1] if len(data) >= 14 else 50
        
        # Calculate confidence based on tech disruptor metrics
        # Tech disruptors like assets that:
        # - Have strong upward momentum
        # - Have increasing volume (growing interest)
        # - Are breaking out above moving averages
        # - Have higher volatility (opportunity for rapid gains)
        
        # Start with neutral confidence
        confidence = 0.0
        
        # Adjust for short-term momentum
        if momentum_1d > 0.05:  # Strong daily momentum
            confidence += 0.3
            momentum_1d_reason = f"Strong 24h momentum of +{momentum_1d*100:.1f}%"
        elif momentum_1d > 0.02:  # Moderate daily momentum
            confidence += 0.2
            momentum_1d_reason = f"Good 24h momentum of +{momentum_1d*100:.1f}%"
        elif momentum_1d < -0.05:  # Strong negative daily momentum
            confidence -= 0.2
            momentum_1d_reason = f"Negative 24h momentum of {momentum_1d*100:.1f}%"
        else:
            momentum_1d_reason = f"24h price change of {momentum_1d*100:.1f}%"
            
        # Adjust for medium-term momentum
        if momentum_7d > 0.15:  # Strong weekly momentum
            confidence += 0.3
            momentum_7d_reason = f"Strong 7-day momentum of +{momentum_7d*100:.1f}%"
        elif momentum_7d > 0.07:  # Moderate weekly momentum
            confidence += 0.2
            momentum_7d_reason = f"Good 7-day momentum of +{momentum_7d*100:.1f}%"
        elif momentum_7d < -0.15:  # Strong negative weekly momentum
            confidence -= 0.2
            momentum_7d_reason = f"Negative 7-day momentum of {momentum_7d*100:.1f}%"
        else:
            momentum_7d_reason = f"7-day price change of {momentum_7d*100:.1f}%"
            
        # Adjust for volume change
        if volume_change > 1.0:  # Volume more than doubled
            confidence += 0.3
            volume_reason = f"Trading volume has surged by {volume_change*100:.1f}%, indicating strong market interest"
        elif volume_change > 0.5:  # Volume increased by 50%+
            confidence += 0.2
            volume_reason = f"Trading volume has increased by {volume_change*100:.1f}%, showing growing market interest"
        elif volume_change < -0.5:  # Volume decreased by 50%+
            confidence -= 0.2
            volume_reason = f"Trading volume has decreased by {abs(volume_change)*100:.1f}%, indicating waning interest"
        else:
            volume_reason = f"Trading volume change of {volume_change*100:.1f}%"
            
        # Adjust for breakout potential
        if price_to_sma > 0.1:  # Strong breakout
            confidence += 0.3
            breakout_reason = f"Price is {price_to_sma*100:.1f}% above 20-period average, indicating a breakout"
        elif price_to_sma > 0.05:  # Moderate breakout
            confidence += 0.2
            breakout_reason = f"Price is {price_to_sma*100:.1f}% above 20-period average, suggesting upward momentum"
        elif price_to_sma < -0.1:  # Below support
            confidence -= 0.2
            breakout_reason = f"Price is {abs(price_to_sma)*100:.1f}% below 20-period average, indicating weakness"
        else:
            breakout_reason = f"Price is {price_to_sma*100:.1f}% relative to 20-period average"
            
        # Adjust for RSI
        if rsi > 70:  # Overbought
            confidence -= 0.1
            rsi_reason = f"RSI of {rsi:.1f} indicates overbought conditions"
        elif rsi < 30:  # Oversold
            confidence += 0.1
            rsi_reason = f"RSI of {rsi:.1f} indicates oversold conditions with potential for rebound"
        elif rsi > 60:  # Strong momentum
            confidence += 0.1
            rsi_reason = f"RSI of {rsi:.1f} indicates strong momentum"
        else:
            rsi_reason = f"RSI of {rsi:.1f}"
            
        # Adjust for volatility (tech disruptors like volatility)
        if volatility > 0.05:  # High volatility
            confidence += 0.1
            volatility_reason = f"High volatility ({volatility:.4f}) presents opportunity for rapid gains"
        elif volatility < 0.01:  # Low volatility
            confidence -= 0.1
            volatility_reason = f"Low volatility ({volatility:.4f}) offers limited opportunity for rapid gains"
        else:
            volatility_reason = f"Moderate volatility ({volatility:.4f})"
            
        # Combine reasons
        reason = f"{momentum_1d_reason}. {momentum_7d_reason}. {volume_reason}. {breakout_reason}. {rsi_reason}. {volatility_reason}."
        
        # Add an Elon Musk style quote for flavor
        if confidence > 0.5:
            reason += f" As I like to say, '{np.random.choice(self.quotes)}'"
            
        # Ensure confidence is within bounds
        confidence = max(-1.0, min(1.0, confidence))
        
        return {
            'confidence': confidence,
            'reason': reason,
            'metrics': {
                'momentum_1d': momentum_1d,
                'momentum_7d': momentum_7d,
                'volume_change': volume_change,
                'price_to_sma': price_to_sma,
                'rsi': rsi,
                'volatility': volatility
            }
        } 