import pandas as pd
import numpy as np
from typing import Dict
import ta
from datetime import datetime, timedelta

from trader.agents.base_agent import BaseAgent

class MacroTrader(BaseAgent):
    """
    Macro Trader agent inspired by Ray Dalio's approach.
    Focuses on macroeconomic trends and global events.
    """
    def __init__(self, name: str = "Ray Dalio AI", timeframe: str = '1d'):
        """
        Initialize the Macro Trader agent.
        
        Args:
            name: Name of the agent
            timeframe: Trading timeframe
        """
        super().__init__(
            name=name,
            personality="Macro Trader - Global perspective, focuses on economic cycles, patient, systematic",
            risk_tolerance=0.5,  # Moderate risk tolerance
            timeframe=timeframe
        )
        
        # Macro trader quotes
        self.quotes = [
            "Markets are reflections of economies.",
            "The biggest mistake investors make is to believe that what happened in the recent past is likely to persist.",
            "The economy is like a machine.",
            "There are few things more important than understanding cause-effect relationships.",
            "The key is to fail well. Learn from your mistakes.",
            "Pain + Reflection = Progress."
        ]
        
        # Set macro trader strategy preferences
        self.set_strategy_preferences({
            'value_focus': 0.6,            # Moderate to strong interest in value
            'trend_following': 0.7,        # Strong interest in trends
            'momentum': 0.5,               # Moderate interest in momentum
            'mean_reversion': 0.4,         # Moderate interest in mean reversion
            'volatility_preference': 0.3,  # Limited interest in volatility
            'speculation': 0.2,            # Limited interest in speculation
            'meme_coin_interest': 0.1,     # Very limited interest in meme coins
            'alt_coin_interest': 0.4       # Moderate interest in alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'long-term',
            'preferred_condition': 'aligned-with-macro-trends',
            'market_approach': 'fundamental-and-cyclical',
            'volatility_view': 'part-of-cycles',
            'crypto_view': 'emerging-asset-class',
            'meme_coin_view': 'mostly-avoid',
            'innovation_view': 'long-term-potential'
        })
        
        # Macro cycle tracking
        self.market_cycle_phase = 'unknown'
        self.last_cycle_update = datetime.now()
        
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
            return {'confidence': 0, 'reason': "Insufficient data for macro analysis"}
            
        # Macro traders focus on long-term trends and market cycles
        
        # Calculate key technical indicators
        current_price = data['close'].iloc[-1]
        
        # Long-term moving averages
        sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
        sma_200 = data['close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else data['close'].rolling(window=len(data)//2).mean().iloc[-1]
        
        # Trend direction and strength
        price_vs_sma50 = (current_price - sma_50) / sma_50
        price_vs_sma200 = (current_price - sma_200) / sma_200
        sma50_vs_sma200 = (sma_50 - sma_200) / sma_200  # Golden/Death cross
        
        # Volume trends
        recent_volume = data['volume'].iloc[-20:].mean()
        earlier_volume = data['volume'].iloc[-50:-20].mean() if len(data) >= 50 else data['volume'].iloc[:len(data)//2].mean()
        volume_trend = (recent_volume - earlier_volume) / earlier_volume if earlier_volume > 0 else 0
        
        # Volatility
        recent_volatility = data['close'].pct_change().iloc[-20:].std() * np.sqrt(252)  # Annualized
        
        # Determine market cycle phase
        self._update_market_cycle(data)
        
        # Calculate confidence based on macro metrics
        # Macro traders like assets that:
        # - Are aligned with the current market cycle phase
        # - Show strong long-term trends
        # - Have healthy volume trends
        # - Are not excessively volatile
        
        # Start with neutral confidence
        confidence = 0.0
        
        # Adjust for market cycle phase
        if self.market_cycle_phase == 'accumulation':
            # Early bull market - good time to buy
            confidence += 0.3
            cycle_reason = f"Market appears to be in accumulation phase, suggesting potential for long-term upside"
        elif self.market_cycle_phase == 'markup':
            # Bull market - hold or add
            confidence += 0.2
            cycle_reason = f"Market appears to be in markup (bull) phase, suggesting continued upside potential"
        elif self.market_cycle_phase == 'distribution':
            # Early bear market - time to reduce
            confidence -= 0.3
            cycle_reason = f"Market appears to be in distribution phase, suggesting potential for downside"
        elif self.market_cycle_phase == 'markdown':
            # Bear market - avoid or short
            confidence -= 0.2
            cycle_reason = f"Market appears to be in markdown (bear) phase, suggesting continued downside risk"
        else:
            cycle_reason = f"Market cycle phase unclear, monitoring for confirmation"
            
        # Adjust for long-term trend
        if price_vs_sma50 > 0.1 and price_vs_sma200 > 0.2 and sma50_vs_sma200 > 0.05:
            # Strong bull trend
            confidence += 0.3
            trend_reason = f"Strong bull trend: price {price_vs_sma50*100:.1f}% above 50-day MA and {price_vs_sma200*100:.1f}% above 200-day MA"
        elif price_vs_sma50 > 0.05 and price_vs_sma200 > 0.1:
            # Moderate bull trend
            confidence += 0.2
            trend_reason = f"Moderate bull trend: price {price_vs_sma50*100:.1f}% above 50-day MA and {price_vs_sma200*100:.1f}% above 200-day MA"
        elif price_vs_sma50 < -0.1 and price_vs_sma200 < -0.2 and sma50_vs_sma200 < -0.05:
            # Strong bear trend
            confidence -= 0.3
            trend_reason = f"Strong bear trend: price {abs(price_vs_sma50)*100:.1f}% below 50-day MA and {abs(price_vs_sma200)*100:.1f}% below 200-day MA"
        elif price_vs_sma50 < -0.05 and price_vs_sma200 < -0.1:
            # Moderate bear trend
            confidence -= 0.2
            trend_reason = f"Moderate bear trend: price {abs(price_vs_sma50)*100:.1f}% below 50-day MA and {abs(price_vs_sma200)*100:.1f}% below 200-day MA"
        elif sma50_vs_sma200 > 0:
            # Bullish MA alignment
            confidence += 0.1
            trend_reason = f"Bullish moving average alignment: 50-day MA {sma50_vs_sma200*100:.1f}% above 200-day MA"
        elif sma50_vs_sma200 < 0:
            # Bearish MA alignment
            confidence -= 0.1
            trend_reason = f"Bearish moving average alignment: 50-day MA {abs(sma50_vs_sma200)*100:.1f}% below 200-day MA"
        else:
            trend_reason = f"No clear long-term trend detected"
            
        # Adjust for volume trend
        if volume_trend > 0.5:
            # Strongly increasing volume
            if confidence > 0:
                confidence += 0.1  # Confirm bullish view
                volume_reason = f"Volume increased by {volume_trend*100:.1f}%, confirming bullish trend"
            elif confidence < 0:
                confidence += 0.05  # Slightly reduce bearish view
                volume_reason = f"Volume increased by {volume_trend*100:.1f}%, suggesting potential trend change"
            else:
                volume_reason = f"Volume increased by {volume_trend*100:.1f}%"
        elif volume_trend < -0.3:
            # Strongly decreasing volume
            if confidence < 0:
                confidence -= 0.1  # Confirm bearish view
                volume_reason = f"Volume decreased by {abs(volume_trend)*100:.1f}%, confirming bearish trend"
            elif confidence > 0:
                confidence -= 0.05  # Slightly reduce bullish view
                volume_reason = f"Volume decreased by {abs(volume_trend)*100:.1f}%, suggesting potential trend change"
            else:
                volume_reason = f"Volume decreased by {abs(volume_trend)*100:.1f}%"
        else:
            volume_reason = f"Volume trend stable at {volume_trend*100:.1f}%"
            
        # Adjust for volatility
        if recent_volatility > 1.0:
            # Extremely high volatility
            confidence *= 0.8  # Reduce confidence due to high risk
            volatility_reason = f"Extremely high volatility ({recent_volatility:.2f} annualized), suggesting elevated risk"
        elif recent_volatility > 0.7:
            # High volatility
            confidence *= 0.9  # Slightly reduce confidence
            volatility_reason = f"High volatility ({recent_volatility:.2f} annualized), suggesting increased risk"
        else:
            volatility_reason = f"Moderate volatility ({recent_volatility:.2f} annualized)"
            
        # Combine reasons
        reason = f"{cycle_reason}. {trend_reason}. {volume_reason}. {volatility_reason}."
        
        # Add a macro trader quote for flavor
        if abs(confidence) > 0.3:
            reason += f" As I often say, '{np.random.choice(self.quotes)}'"
            
        # Ensure confidence is within bounds
        confidence = max(-1.0, min(1.0, confidence))
        
        return {
            'confidence': confidence,
            'reason': reason,
            'metrics': {
                'price_vs_sma50': price_vs_sma50,
                'price_vs_sma200': price_vs_sma200,
                'sma50_vs_sma200': sma50_vs_sma200,
                'volume_trend': volume_trend,
                'volatility': recent_volatility,
                'market_cycle': self.market_cycle_phase
            }
        }
        
    def _update_market_cycle(self, data: pd.DataFrame) -> None:
        """
        Update the market cycle phase assessment.
        
        Args:
            data: DataFrame with market data
        """
        # Only update periodically (once per day)
        if (datetime.now() - self.last_cycle_update) < timedelta(days=1):
            return
            
        # Simple market cycle detection based on price action and volume
        if len(data) < 100:
            self.market_cycle_phase = 'unknown'
            return
            
        # Get recent price action
        current_price = data['close'].iloc[-1]
        recent_low = data['low'].iloc[-50:].min()
        recent_high = data['high'].iloc[-50:].max()
        
        # Get moving averages
        sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
        sma_200 = data['close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else data['close'].rolling(window=len(data)//2).mean().iloc[-1]
        
        # Get volume trends
        recent_volume = data['volume'].iloc[-20:].mean()
        earlier_volume = data['volume'].iloc[-50:-20].mean()
        volume_trend = (recent_volume - earlier_volume) / earlier_volume if earlier_volume > 0 else 0
        
        # Calculate price position within recent range
        if recent_high > recent_low:
            price_position = (current_price - recent_low) / (recent_high - recent_low)
        else:
            price_position = 0.5
            
        # Determine market cycle phase
        if current_price < sma_50 and current_price < sma_200 and price_position < 0.3 and volume_trend > 0.2:
            # Price below MAs, near recent lows, but volume increasing - potential accumulation
            self.market_cycle_phase = 'accumulation'
        elif current_price > sma_50 and price_position > 0.3 and price_position < 0.7 and volume_trend > 0:
            # Price above 50 MA, in middle of range, volume stable or increasing - likely markup (bull)
            self.market_cycle_phase = 'markup'
        elif current_price > sma_50 and current_price > sma_200 and price_position > 0.7 and volume_trend < 0:
            # Price above MAs, near recent highs, but volume decreasing - potential distribution
            self.market_cycle_phase = 'distribution'
        elif current_price < sma_50 and price_position < 0.7 and volume_trend < 0:
            # Price below 50 MA, not at highs, volume decreasing - likely markdown (bear)
            self.market_cycle_phase = 'markdown'
        else:
            # Not clear
            self.market_cycle_phase = 'unknown'
            
        self.last_cycle_update = datetime.now() 