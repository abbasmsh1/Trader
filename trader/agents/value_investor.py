import pandas as pd
import numpy as np
from typing import Dict
import ta

from trader.agents.base_agent import BaseAgent

class ValueInvestor(BaseAgent):
    """
    Value Investor agent inspired by Warren Buffett's principles.
    Uses fundamental analysis and looks for undervalued assets with strong potential.
    """
    def __init__(self, name: str = "Warren Buffett AI", timeframe: str = '1d'):
        """
        Initialize the Value Investor agent.
        
        Args:
            name: Name of the agent
            timeframe: Trading timeframe
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
            
        # Value investors look for assets that are undervalued relative to their intrinsic value
        # In crypto, we'll use some proxies for "value" since traditional value metrics don't apply
        
        # 1. Look for assets that have fallen significantly from their recent highs (potential value)
        current_price = data['close'].iloc[-1]
        max_price = data['high'].max()
        price_from_max = (current_price - max_price) / max_price  # Negative value = below max
        
        # 2. Check if volume is increasing (interest is growing)
        recent_volume = data['volume'].iloc[-5:].mean()
        earlier_volume = data['volume'].iloc[-20:-5].mean()
        volume_change = (recent_volume - earlier_volume) / earlier_volume if earlier_volume > 0 else 0
        
        # 3. Check for mean reversion potential
        sma_50 = data['close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else data['close'].mean()
        price_to_sma = (current_price - sma_50) / sma_50
        
        # 4. Check for stability (value investors prefer stability)
        volatility = data['close'].pct_change().std()
        
        # Calculate confidence based on value metrics
        # Value investors like assets that:
        # - Have fallen from highs but show signs of stabilization
        # - Have increasing volume (growing interest)
        # - Are below their long-term moving averages
        # - Have relatively low volatility
        
        # Start with neutral confidence
        confidence = 0.0
        
        # Adjust for price from maximum (value opportunity if price has fallen)
        if price_from_max < -0.5:  # More than 50% below all-time high
            confidence += 0.4  # Major value opportunity
            value_reason = f"Price is {abs(price_from_max)*100:.1f}% below recent high, presenting significant value"
        elif price_from_max < -0.3:  # 30-50% below all-time high
            confidence += 0.3  # Good value opportunity
            value_reason = f"Price is {abs(price_from_max)*100:.1f}% below recent high, presenting good value"
        elif price_from_max < -0.15:  # 15-30% below all-time high
            confidence += 0.2  # Moderate value opportunity
            value_reason = f"Price is {abs(price_from_max)*100:.1f}% below recent high, presenting moderate value"
        elif price_from_max > 0:  # Above all-time high
            confidence -= 0.3  # Overvalued
            value_reason = f"Price is at or near all-time high, suggesting potential overvaluation"
        else:
            value_reason = f"Price is {abs(price_from_max)*100:.1f}% below recent high"
            
        # Adjust for volume change
        if volume_change > 0.5:  # Volume increasing significantly
            confidence += 0.2
            volume_reason = f"Trading volume has increased by {volume_change*100:.1f}%, indicating growing interest"
        elif volume_change > 0.2:  # Volume increasing moderately
            confidence += 0.1
            volume_reason = f"Trading volume has increased by {volume_change*100:.1f}%"
        elif volume_change < -0.3:  # Volume decreasing significantly
            confidence -= 0.1
            volume_reason = f"Trading volume has decreased by {abs(volume_change)*100:.1f}%, indicating waning interest"
        else:
            volume_reason = f"Trading volume is relatively stable"
            
        # Adjust for price relative to SMA
        if price_to_sma < -0.2:  # Significantly below SMA
            confidence += 0.2
            sma_reason = f"Price is {abs(price_to_sma)*100:.1f}% below 50-period average, suggesting undervaluation"
        elif price_to_sma > 0.2:  # Significantly above SMA
            confidence -= 0.2
            sma_reason = f"Price is {price_to_sma*100:.1f}% above 50-period average, suggesting potential overvaluation"
        else:
            sma_reason = f"Price is near its 50-period average"
            
        # Adjust for volatility
        if volatility > 0.05:  # High volatility
            confidence -= 0.2
            volatility_reason = f"High price volatility ({volatility:.4f}) increases risk"
        elif volatility < 0.02:  # Low volatility
            confidence += 0.1
            volatility_reason = f"Low price volatility ({volatility:.4f}) suggests stability"
        else:
            volatility_reason = f"Moderate price volatility ({volatility:.4f})"
            
        # Combine reasons
        reason = f"{value_reason}. {volume_reason}. {sma_reason}. {volatility_reason}."
        
        # Add a Warren Buffett quote for flavor
        if confidence > 0.5:
            reason += f" As I always say, '{np.random.choice(self.quotes)}'"
            
        # Ensure confidence is within bounds
        confidence = max(-1.0, min(1.0, confidence))
        
        return {
            'confidence': confidence,
            'reason': reason,
            'metrics': {
                'price_from_max': price_from_max,
                'volume_change': volume_change,
                'price_to_sma': price_to_sma,
                'volatility': volatility
            }
        } 