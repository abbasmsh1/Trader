import pandas as pd
import numpy as np
from typing import Dict
import ta

from trader.agents.base_agent import BaseAgent

class ContrarianTrader(BaseAgent):
    """
    Contrarian Trader agent inspired by Michael Burry's approach.
    Looks for market inefficiencies and trades against the crowd.
    """
    def __init__(self, name: str = "Michael Burry AI", timeframe: str = '1d'):
        """
        Initialize the Contrarian Trader agent.
        
        Args:
            name: Name of the agent
            timeframe: Trading timeframe
        """
        super().__init__(
            name=name,
            personality="Contrarian Trader - Skeptical, independent thinker, trades against the crowd, seeks market inefficiencies",
            risk_tolerance=0.7,  # High risk tolerance
            timeframe=timeframe
        )
        
        # Contrarian quotes
        self.quotes = [
            "Be fearful when others are greedy, and greedy when others are fearful.",
            "The crowd is almost always wrong at key turning points.",
            "The time to buy is when there's blood in the streets.",
            "Markets tend to overreact in both directions.",
            "The herd mentality leads to market extremes.",
            "The majority is often wrong; the minority sometimes right."
        ]
        
        # Set contrarian strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.7,            # Strong interest in value
            'trend_following': 0.1,        # Very limited interest in trends
            'momentum': 0.1,               # Very limited interest in momentum
            'mean_reversion': 0.9,         # Very strong interest in mean reversion
            'volatility_preference': 0.6,  # Moderate to high interest in volatility
            'speculation': 0.4,            # Moderate interest in speculation
            'meme_coin_interest': 0.2,     # Limited interest in meme coins
            'alt_coin_interest': 0.5       # Moderate interest in alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'medium-to-long-term',
            'preferred_condition': 'market-extremes',
            'market_approach': 'contrarian',
            'volatility_view': 'opportunity-for-profit',
            'crypto_view': 'skeptical-but-opportunistic',
            'meme_coin_view': 'mostly-avoid',
            'innovation_view': 'value-based-assessment'
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
            return {'confidence': 0, 'reason': "Insufficient data for contrarian analysis"}
            
        # Contrarians look for extreme market conditions and trade against them
        
        # Calculate key technical indicators
        current_price = data['close'].iloc[-1]
        
        # RSI for overbought/oversold conditions
        rsi = ta.momentum.RSIIndicator(data['close']).rsi().iloc[-1]
        
        # Bollinger Bands for price extremes
        bb = ta.volatility.BollingerBands(data['close'])
        upper_band = bb.bollinger_hband().iloc[-1]
        lower_band = bb.bollinger_lband().iloc[-1]
        bb_width = (upper_band - lower_band) / bb.bollinger_mavg().iloc[-1]
        
        # Price distance from bands
        price_to_upper = (upper_band - current_price) / current_price
        price_to_lower = (current_price - lower_band) / current_price
        
        # Recent price action
        price_change_5d = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5] if len(data) >= 5 else 0
        price_change_20d = (current_price - data['close'].iloc[-20]) / data['close'].iloc[-20] if len(data) >= 20 else 0
        
        # Volume analysis
        recent_volume = data['volume'].iloc[-5:].mean()
        avg_volume = data['volume'].iloc[-20:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate confidence based on contrarian metrics
        # Contrarians like to:
        # - Buy when assets are oversold (low RSI)
        # - Sell when assets are overbought (high RSI)
        # - Look for extreme price movements to fade
        # - Pay attention to unusual volume as sign of potential reversal
        
        # Start with neutral confidence
        confidence = 0.0
        
        # Adjust for RSI extremes (contrarians buy oversold, sell overbought)
        if rsi >= 80:
            # Extremely overbought
            confidence -= 0.5
            rsi_reason = f"Extremely overbought conditions: RSI of {rsi:.1f} suggests potential for significant downside"
        elif rsi >= 70:
            # Overbought
            confidence -= 0.3
            rsi_reason = f"Overbought conditions: RSI of {rsi:.1f} suggests potential for downside"
        elif rsi <= 20:
            # Extremely oversold
            confidence += 0.5
            rsi_reason = f"Extremely oversold conditions: RSI of {rsi:.1f} suggests potential for significant upside"
        elif rsi <= 30:
            # Oversold
            confidence += 0.3
            rsi_reason = f"Oversold conditions: RSI of {rsi:.1f} suggests potential for upside"
        else:
            rsi_reason = f"Neutral RSI of {rsi:.1f}, no extreme conditions detected"
            
        # Adjust for Bollinger Band extremes
        if price_to_upper < 0.01:
            # Price at or above upper band
            confidence -= 0.3
            bb_reason = f"Price at upper Bollinger Band, suggesting overbought conditions"
        elif price_to_lower < 0.01:
            # Price at or below lower band
            confidence += 0.3
            bb_reason = f"Price at lower Bollinger Band, suggesting oversold conditions"
        elif bb_width > 0.1:
            # Wide Bollinger Bands (high volatility)
            if price_to_upper < 0.05:
                confidence -= 0.2
                bb_reason = f"High volatility with price near upper band, potential reversal point"
            elif price_to_lower < 0.05:
                confidence += 0.2
                bb_reason = f"High volatility with price near lower band, potential reversal point"
            else:
                bb_reason = f"High volatility but price not at extremes"
        else:
            bb_reason = f"Price within normal Bollinger Band range"
            
        # Adjust for recent price action (fade extreme moves)
        if price_change_5d > 0.2:
            # Strong short-term rally
            confidence -= 0.3
            price_action_reason = f"Price surged {price_change_5d*100:.1f}% in 5 days, suggesting potential overextension"
        elif price_change_5d < -0.2:
            # Strong short-term drop
            confidence += 0.3
            price_action_reason = f"Price dropped {abs(price_change_5d)*100:.1f}% in 5 days, suggesting potential overselling"
        elif price_change_20d > 0.5:
            # Strong medium-term rally
            confidence -= 0.2
            price_action_reason = f"Price surged {price_change_20d*100:.1f}% in 20 days, suggesting potential overextension"
        elif price_change_20d < -0.5:
            # Strong medium-term drop
            confidence += 0.2
            price_action_reason = f"Price dropped {abs(price_change_20d)*100:.1f}% in 20 days, suggesting potential overselling"
        else:
            price_action_reason = f"No extreme price action detected"
            
        # Adjust for unusual volume
        if volume_ratio > 3.0:
            # Extremely high volume
            if confidence > 0:
                confidence += 0.1  # Confirm bullish view
                volume_reason = f"Volume {volume_ratio:.1f}x above average, confirming potential reversal from oversold conditions"
            elif confidence < 0:
                confidence -= 0.1  # Confirm bearish view
                volume_reason = f"Volume {volume_ratio:.1f}x above average, confirming potential reversal from overbought conditions"
            else:
                volume_reason = f"Volume {volume_ratio:.1f}x above average, suggesting potential market turning point"
        elif volume_ratio > 2.0:
            # High volume
            if confidence > 0:
                confidence += 0.05  # Slightly confirm bullish view
                volume_reason = f"Volume {volume_ratio:.1f}x above average, supporting potential reversal from oversold conditions"
            elif confidence < 0:
                confidence -= 0.05  # Slightly confirm bearish view
                volume_reason = f"Volume {volume_ratio:.1f}x above average, supporting potential reversal from overbought conditions"
            else:
                volume_reason = f"Volume {volume_ratio:.1f}x above average, suggesting increased market interest"
        else:
            volume_reason = f"Normal trading volume at {volume_ratio:.1f}x average"
            
        # Combine reasons
        reason = f"{rsi_reason}. {bb_reason}. {price_action_reason}. {volume_reason}."
        
        # Add a contrarian quote for flavor
        if abs(confidence) > 0.3:
            reason += f" As contrarians say, '{np.random.choice(self.quotes)}'"
            
        # Ensure confidence is within bounds
        confidence = max(-1.0, min(1.0, confidence))
        
        return {
            'confidence': confidence,
            'reason': reason,
            'metrics': {
                'rsi': rsi,
                'price_to_upper_band': price_to_upper,
                'price_to_lower_band': price_to_lower,
                'bb_width': bb_width,
                'price_change_5d': price_change_5d,
                'price_change_20d': price_change_20d,
                'volume_ratio': volume_ratio
            }
        } 