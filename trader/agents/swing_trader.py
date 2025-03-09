import pandas as pd
import numpy as np
from typing import Dict
import ta

from trader.agents.base_agent import BaseAgent

class SwingTrader(BaseAgent):
    """
    Swing Trader agent inspired by Jesse Livermore's approach.
    Focuses on capturing medium-term price swings using technical analysis.
    """
    def __init__(self, name: str = "Jesse Livermore AI", timeframe: str = '4h'):
        """
        Initialize the Swing Trader agent.
        
        Args:
            name: Name of the agent
            timeframe: Trading timeframe
        """
        super().__init__(
            name=name,
            personality="Swing Trader - Momentum-based, short to medium-term focused, technical analysis driven",
            risk_tolerance=0.7,  # High risk tolerance
            timeframe=timeframe
        )
        
        # Swing trader quotes
        self.quotes = [
            "The market does not beat them. They beat themselves, because though they have brains they cannot sit tight.",
            "It takes time to make money.",
            "There is a time for all things, but I didn't know it. And that is precisely what beats so many men in Wall Street who are very far from being in the main sucker class.",
            "Play the market only when all factors are in your favor. No person can play the market all the time and win.",
            "The game of speculation is the most uniformly fascinating game in the world. But it is not a game for the stupid, the mentally lazy, the person of inferior emotional balance, or the get-rich-quick adventurer. They will die poor.",
            "Money is made by sitting, not trading."
        ]
        
        # Set swing trader strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.3,            # Limited interest in value
            'trend_following': 0.6,        # Moderate to strong interest in trends
            'momentum': 0.8,               # Strong interest in momentum
            'mean_reversion': 0.5,         # Moderate interest in mean reversion
            'volatility_preference': 0.7,  # Strong interest in volatility
            'speculation': 0.6,            # Moderate to strong interest in speculation
            'meme_coin_interest': 0.4,     # Moderate interest in meme coins
            'alt_coin_interest': 0.6       # Moderate to strong interest in alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'short-to-medium-term',
            'preferred_condition': 'volatile-with-clear-direction',
            'market_approach': 'technical-and-momentum',
            'volatility_view': 'opportunity-for-swings',
            'crypto_view': 'trading-vehicle',
            'meme_coin_view': 'trading-opportunities',
            'innovation_view': 'momentum-based-assessment'
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
        if data.empty or len(data) < 30:
            return {'confidence': 0, 'reason': "Insufficient data for swing analysis"}
            
        # Swing traders focus on capturing medium-term price swings
        
        # Calculate key technical indicators
        current_price = data['close'].iloc[-1]
        
        # Moving averages
        sma_10 = data['close'].rolling(window=10).mean().iloc[-1]
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        
        # Trend direction and strength
        price_vs_sma10 = (current_price - sma_10) / sma_10
        price_vs_sma20 = (current_price - sma_20) / sma_20
        sma10_vs_sma20 = (sma_10 - sma_20) / sma_20  # Moving average crossover
        
        # Calculate RSI for momentum
        rsi = ta.momentum.RSIIndicator(data['close']).rsi().iloc[-1]
        
        # Calculate Stochastic for overbought/oversold
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        stoch_k = stoch.stoch().iloc[-1]
        stoch_d = stoch.stoch_signal().iloc[-1]
        
        # Calculate MACD for trend confirmation
        macd = ta.trend.MACD(data['close'])
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        macd_histogram = macd.macd_diff().iloc[-1]
        
        # Calculate Bollinger Bands for volatility and price extremes
        bb = ta.volatility.BollingerBands(data['close'])
        upper_band = bb.bollinger_hband().iloc[-1]
        lower_band = bb.bollinger_lband().iloc[-1]
        bb_width = (upper_band - lower_band) / bb.bollinger_mavg().iloc[-1]
        
        # Price distance from bands
        price_to_upper = (upper_band - current_price) / current_price
        price_to_lower = (current_price - lower_band) / current_price
        
        # Recent price action
        price_change_3d = (current_price - data['close'].iloc[-3]) / data['close'].iloc[-3] if len(data) >= 3 else 0
        
        # Volume analysis
        recent_volume = data['volume'].iloc[-3:].mean()
        avg_volume = data['volume'].iloc[-10:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate confidence based on swing trading metrics
        # Swing traders like to:
        # - Buy when momentum is turning up from oversold conditions
        # - Sell when momentum is turning down from overbought conditions
        # - Look for confirmation from multiple indicators
        # - Pay attention to volume as confirmation
        
        # Start with neutral confidence
        confidence = 0.0
        
        # Determine if we're in a potential swing long or short setup
        swing_long_setup = (
            rsi < 40 and rsi > rsi.shift(1) and  # RSI turning up from oversold
            stoch_k < 30 and stoch_k > stoch_k.shift(1) and  # Stochastic turning up from oversold
            price_to_lower < 0.05 and  # Price near lower Bollinger Band
            macd_histogram > macd_histogram.shift(1)  # MACD histogram turning up
        )
        
        swing_short_setup = (
            rsi > 60 and rsi < rsi.shift(1) and  # RSI turning down from overbought
            stoch_k > 70 and stoch_k < stoch_k.shift(1) and  # Stochastic turning down from overbought
            price_to_upper < 0.05 and  # Price near upper Bollinger Band
            macd_histogram < macd_histogram.shift(1)  # MACD histogram turning down
        )
        
        # Adjust for swing setups
        if swing_long_setup:
            confidence += 0.5
            setup_reason = "Strong swing long setup: RSI turning up from oversold, Stochastic turning up from oversold, price near lower Bollinger Band, MACD histogram turning up"
        elif swing_short_setup:
            confidence -= 0.5
            setup_reason = "Strong swing short setup: RSI turning down from overbought, Stochastic turning down from overbought, price near upper Bollinger Band, MACD histogram turning down"
        else:
            # Check individual components
            # RSI
            if rsi < 30:
                confidence += 0.2
                rsi_reason = f"Oversold RSI of {rsi:.1f} suggests potential for upward swing"
            elif rsi > 70:
                confidence -= 0.2
                rsi_reason = f"Overbought RSI of {rsi:.1f} suggests potential for downward swing"
            elif rsi < 40 and rsi > rsi.shift(1):
                confidence += 0.1
                rsi_reason = f"RSI of {rsi:.1f} turning up from oversold region"
            elif rsi > 60 and rsi < rsi.shift(1):
                confidence -= 0.1
                rsi_reason = f"RSI of {rsi:.1f} turning down from overbought region"
            else:
                rsi_reason = f"Neutral RSI of {rsi:.1f}"
                
            # Stochastic
            if stoch_k < 20 and stoch_k > stoch_d:
                confidence += 0.2
                stoch_reason = f"Stochastic ({stoch_k:.1f}) crossing above signal in oversold region"
            elif stoch_k > 80 and stoch_k < stoch_d:
                confidence -= 0.2
                stoch_reason = f"Stochastic ({stoch_k:.1f}) crossing below signal in overbought region"
            elif stoch_k < 30:
                confidence += 0.1
                stoch_reason = f"Oversold Stochastic of {stoch_k:.1f}"
            elif stoch_k > 70:
                confidence -= 0.1
                stoch_reason = f"Overbought Stochastic of {stoch_k:.1f}"
            else:
                stoch_reason = f"Neutral Stochastic of {stoch_k:.1f}"
                
            # MACD
            if macd_line > signal_line and macd_histogram > 0 and macd_histogram > macd_histogram.shift(1):
                confidence += 0.2
                macd_reason = f"MACD ({macd_line:.4f}) above signal with increasing histogram, confirming upward momentum"
            elif macd_line < signal_line and macd_histogram < 0 and macd_histogram < macd_histogram.shift(1):
                confidence -= 0.2
                macd_reason = f"MACD ({macd_line:.4f}) below signal with decreasing histogram, confirming downward momentum"
            elif macd_line > signal_line:
                confidence += 0.1
                macd_reason = f"MACD ({macd_line:.4f}) above signal line ({signal_line:.4f})"
            elif macd_line < signal_line:
                confidence -= 0.1
                macd_reason = f"MACD ({macd_line:.4f}) below signal line ({signal_line:.4f})"
            else:
                macd_reason = f"Neutral MACD ({macd_line:.4f})"
                
            # Bollinger Bands
            if price_to_lower < 0.02:
                confidence += 0.2
                bb_reason = f"Price at lower Bollinger Band, potential for upward swing"
            elif price_to_upper < 0.02:
                confidence -= 0.2
                bb_reason = f"Price at upper Bollinger Band, potential for downward swing"
            elif bb_width > 0.1 and price_change_3d > 0.05:
                confidence += 0.1
                bb_reason = f"High volatility with recent upward price action, potential for continued upward swing"
            elif bb_width > 0.1 and price_change_3d < -0.05:
                confidence -= 0.1
                bb_reason = f"High volatility with recent downward price action, potential for continued downward swing"
            else:
                bb_reason = f"Price within normal Bollinger Band range"
                
            # Combine individual reasons
            setup_reason = f"{rsi_reason}. {stoch_reason}. {macd_reason}. {bb_reason}."
            
        # Adjust for moving average alignment
        if price_vs_sma10 > 0 and price_vs_sma20 > 0 and sma10_vs_sma20 > 0:
            # Bullish MA alignment
            confidence += 0.2
            ma_reason = f"Bullish moving average alignment: price above both 10 and 20 period MAs"
        elif price_vs_sma10 < 0 and price_vs_sma20 < 0 and sma10_vs_sma20 < 0:
            # Bearish MA alignment
            confidence -= 0.2
            ma_reason = f"Bearish moving average alignment: price below both 10 and 20 period MAs"
        elif price_vs_sma10 > 0 and sma10_vs_sma20 > 0:
            # Emerging bullish trend
            confidence += 0.1
            ma_reason = f"Emerging bullish trend: price above 10 period MA, 10 period MA above 20 period MA"
        elif price_vs_sma10 < 0 and sma10_vs_sma20 < 0:
            # Emerging bearish trend
            confidence -= 0.1
            ma_reason = f"Emerging bearish trend: price below 10 period MA, 10 period MA below 20 period MA"
        else:
            ma_reason = f"Mixed moving average signals"
            
        # Adjust for volume
        if volume_ratio > 2.0 and confidence > 0:
            confidence += 0.1
            volume_reason = f"Volume {volume_ratio:.1f}x above average, confirming bullish momentum"
        elif volume_ratio > 2.0 and confidence < 0:
            confidence -= 0.1
            volume_reason = f"Volume {volume_ratio:.1f}x above average, confirming bearish momentum"
        else:
            volume_reason = f"Volume at {volume_ratio:.1f}x average"
            
        # Combine reasons
        reason = f"{setup_reason} {ma_reason}. {volume_reason}."
        
        # Add a swing trader quote for flavor
        if abs(confidence) > 0.3:
            reason += f" As Jesse Livermore would say, '{np.random.choice(self.quotes)}'"
            
        # Ensure confidence is within bounds
        confidence = max(-1.0, min(1.0, confidence))
        
        return {
            'confidence': confidence,
            'reason': reason,
            'metrics': {
                'rsi': rsi,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'macd_histogram': macd_histogram,
                'price_to_upper_band': price_to_upper,
                'price_to_lower_band': price_to_lower,
                'bb_width': bb_width,
                'price_change_3d': price_change_3d,
                'volume_ratio': volume_ratio
            }
        } 