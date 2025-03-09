import pandas as pd
import numpy as np
from typing import Dict
import ta

from trader.agents.base_agent import BaseAgent

class TrendFollower(BaseAgent):
    """
    Trend Follower agent inspired by systematic trend-following strategies.
    Uses technical analysis to identify and follow market trends.
    """
    def __init__(self, name: str = "Trend Follower AI", timeframe: str = '1d'):
        """
        Initialize the Trend Follower agent.
        
        Args:
            name: Name of the agent
            timeframe: Trading timeframe
        """
        super().__init__(
            name=name,
            personality="Trend Follower - Systematic, disciplined, follows market trends, technical analysis focused",
            risk_tolerance=0.6,  # Moderate to high risk tolerance
            timeframe=timeframe
        )
        
        # Trend follower quotes
        self.quotes = [
            "The trend is your friend until it ends.",
            "Cut your losses short and let your winners run.",
            "Don't fight the tape.",
            "Trade in the direction of the trend.",
            "Markets can remain in a trend much longer than you can remain solvent betting against it.",
            "In a trending market, the most profitable trades are trend-following trades."
        ]
        
        # Set trend follower strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.2,            # Limited interest in traditional value
            'trend_following': 0.9,        # Very strong interest in trends
            'momentum': 0.8,               # Strong interest in momentum
            'mean_reversion': 0.1,         # Very limited interest in mean reversion
            'volatility_preference': 0.5,  # Moderate interest in volatility
            'speculation': 0.3,            # Limited interest in speculation
            'meme_coin_interest': 0.3,     # Limited interest in meme coins
            'alt_coin_interest': 0.5       # Moderate interest in alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'medium-term',
            'preferred_condition': 'trending-markets',
            'market_approach': 'technical-driven',
            'volatility_view': 'accepts-volatility',
            'crypto_view': 'trend-sensitive',
            'meme_coin_view': 'follows-momentum',
            'innovation_view': 'trend-dependent'
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
        if data.empty or len(data) < 50:
            return {'confidence': 0, 'reason': "Insufficient data for trend analysis"}
            
        # Trend followers focus on identifying and following established trends
        
        # Calculate key technical indicators
        current_price = data['close'].iloc[-1]
        
        # Moving averages
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
        
        # Trend direction and strength
        price_vs_sma20 = (current_price - sma_20) / sma_20
        price_vs_sma50 = (current_price - sma_50) / sma_50
        sma20_vs_sma50 = (sma_20 - sma_50) / sma_50  # Moving average crossover
        
        # Calculate MACD for trend confirmation
        macd = ta.trend.MACD(data['close'])
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        macd_histogram = macd.macd_diff().iloc[-1]
        
        # Calculate RSI for overbought/oversold conditions
        rsi = ta.momentum.RSIIndicator(data['close']).rsi().iloc[-1]
        
        # Calculate ADX for trend strength
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close']).adx().iloc[-1]
        
        # Calculate confidence based on trend metrics
        # Trend followers like assets that:
        # - Are in a clear trend (price above/below moving averages)
        # - Have strong trend momentum (MACD confirmation)
        # - Have strong trend strength (high ADX)
        # - Are not extremely overbought/oversold (RSI not extreme)
        
        # Start with neutral confidence
        confidence = 0.0
        
        # Determine trend direction and strength
        if price_vs_sma20 > 0 and price_vs_sma50 > 0 and sma20_vs_sma50 > 0:
            # Strong uptrend
            trend_direction = "uptrend"
            trend_strength = min(1.0, (price_vs_sma20 + price_vs_sma50 + sma20_vs_sma50) / 3 * 10)
            confidence += 0.3 + trend_strength * 0.3  # Up to +0.6 for very strong uptrend
            trend_reason = f"Strong uptrend: price above both 20 and 50 period moving averages by {price_vs_sma20*100:.1f}% and {price_vs_sma50*100:.1f}%"
        elif price_vs_sma20 < 0 and price_vs_sma50 < 0 and sma20_vs_sma50 < 0:
            # Strong downtrend
            trend_direction = "downtrend"
            trend_strength = min(1.0, (abs(price_vs_sma20) + abs(price_vs_sma50) + abs(sma20_vs_sma50)) / 3 * 10)
            confidence -= 0.3 + trend_strength * 0.3  # Up to -0.6 for very strong downtrend
            trend_reason = f"Strong downtrend: price below both 20 and 50 period moving averages by {abs(price_vs_sma20)*100:.1f}% and {abs(price_vs_sma50)*100:.1f}%"
        elif price_vs_sma20 > 0 and sma20_vs_sma50 > 0:
            # Emerging uptrend
            trend_direction = "emerging uptrend"
            confidence += 0.2
            trend_reason = f"Emerging uptrend: price above 20 period moving average by {price_vs_sma20*100:.1f}%, 20MA crossing above 50MA"
        elif price_vs_sma20 < 0 and sma20_vs_sma50 < 0:
            # Emerging downtrend
            trend_direction = "emerging downtrend"
            confidence -= 0.2
            trend_reason = f"Emerging downtrend: price below 20 period moving average by {abs(price_vs_sma20)*100:.1f}%, 20MA crossing below 50MA"
        elif price_vs_sma20 > 0 and price_vs_sma50 > 0:
            # Moderate uptrend
            trend_direction = "moderate uptrend"
            confidence += 0.1
            trend_reason = f"Moderate uptrend: price above both moving averages but 20MA below 50MA"
        elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
            # Moderate downtrend
            trend_direction = "moderate downtrend"
            confidence -= 0.1
            trend_reason = f"Moderate downtrend: price below both moving averages but 20MA above 50MA"
        else:
            # No clear trend
            trend_direction = "sideways"
            confidence += 0.0
            trend_reason = f"No clear trend: mixed signals from moving averages"
            
        # Adjust for MACD
        if macd_line > signal_line and macd_histogram > 0:
            # Bullish MACD
            if trend_direction in ["uptrend", "emerging uptrend", "moderate uptrend"]:
                # Confirming uptrend
                confidence += 0.2
                macd_reason = f"MACD confirms uptrend: MACD ({macd_line:.4f}) above signal line ({signal_line:.4f})"
            else:
                # Contradicting downtrend
                confidence += 0.1
                macd_reason = f"MACD showing potential trend reversal: MACD ({macd_line:.4f}) above signal line ({signal_line:.4f})"
        elif macd_line < signal_line and macd_histogram < 0:
            # Bearish MACD
            if trend_direction in ["downtrend", "emerging downtrend", "moderate downtrend"]:
                # Confirming downtrend
                confidence -= 0.2
                macd_reason = f"MACD confirms downtrend: MACD ({macd_line:.4f}) below signal line ({signal_line:.4f})"
            else:
                # Contradicting uptrend
                confidence -= 0.1
                macd_reason = f"MACD showing potential trend reversal: MACD ({macd_line:.4f}) below signal line ({signal_line:.4f})"
        else:
            macd_reason = f"MACD ({macd_line:.4f}) near signal line ({signal_line:.4f}), no clear signal"
            
        # Adjust for ADX (trend strength)
        if adx > 30:
            # Strong trend
            if trend_direction in ["uptrend", "emerging uptrend", "moderate uptrend"]:
                confidence += 0.2
                adx_reason = f"Strong trend strength: ADX of {adx:.1f} confirms strong uptrend"
            elif trend_direction in ["downtrend", "emerging downtrend", "moderate downtrend"]:
                confidence -= 0.2
                adx_reason = f"Strong trend strength: ADX of {adx:.1f} confirms strong downtrend"
            else:
                adx_reason = f"Strong trend strength: ADX of {adx:.1f}, but no clear direction"
        elif adx > 20:
            # Moderate trend
            if trend_direction in ["uptrend", "emerging uptrend", "moderate uptrend"]:
                confidence += 0.1
                adx_reason = f"Moderate trend strength: ADX of {adx:.1f} confirms uptrend"
            elif trend_direction in ["downtrend", "emerging downtrend", "moderate downtrend"]:
                confidence -= 0.1
                adx_reason = f"Moderate trend strength: ADX of {adx:.1f} confirms downtrend"
            else:
                adx_reason = f"Moderate trend strength: ADX of {adx:.1f}, but no clear direction"
        else:
            # Weak trend
            confidence *= 0.8  # Reduce confidence due to weak trend
            adx_reason = f"Weak trend strength: ADX of {adx:.1f} indicates potential ranging market"
            
        # Adjust for RSI (avoid extreme conditions)
        if rsi > 70:
            # Overbought
            confidence -= 0.2
            rsi_reason = f"Overbought conditions: RSI of {rsi:.1f} indicates potential reversal risk"
        elif rsi < 30:
            # Oversold
            confidence += 0.2
            rsi_reason = f"Oversold conditions: RSI of {rsi:.1f} indicates potential reversal opportunity"
        elif rsi > 60 and trend_direction in ["uptrend", "emerging uptrend", "moderate uptrend"]:
            # Strong momentum in uptrend
            confidence += 0.1
            rsi_reason = f"Strong momentum: RSI of {rsi:.1f} confirms uptrend"
        elif rsi < 40 and trend_direction in ["downtrend", "emerging downtrend", "moderate downtrend"]:
            # Strong momentum in downtrend
            confidence -= 0.1
            rsi_reason = f"Strong momentum: RSI of {rsi:.1f} confirms downtrend"
        else:
            rsi_reason = f"Neutral momentum: RSI of {rsi:.1f}"
            
        # Combine reasons
        reason = f"{trend_reason}. {macd_reason}. {adx_reason}. {rsi_reason}."
        
        # Add a trend follower quote for flavor
        if abs(confidence) > 0.5:
            reason += f" Remember, '{np.random.choice(self.quotes)}'"
            
        # Ensure confidence is within bounds
        confidence = max(-1.0, min(1.0, confidence))
        
        return {
            'confidence': confidence,
            'reason': reason,
            'metrics': {
                'price_vs_sma20': price_vs_sma20,
                'price_vs_sma50': price_vs_sma50,
                'sma20_vs_sma50': sma20_vs_sma50,
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': macd_histogram,
                'rsi': rsi,
                'adx': adx
            }
        } 