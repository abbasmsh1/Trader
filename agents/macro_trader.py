import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class MacroTrader(BaseAgent):
    def __init__(self, name: str = "Ray Dalio AI", timeframe: str = '1d'):
        """
        Initialize the Macro Trader agent inspired by Ray Dalio.
        Known for systematic trading and global macro analysis.
        """
        super().__init__(
            name=name,
            personality="Macro - Systematic, global perspective, risk-parity focused, all-weather",
            risk_tolerance=0.6,  # Moderate risk tolerance with strong risk management
            timeframe=timeframe
        )
        
        # Ray Dalio's famous quotes
        self.quotes = [
            "The biggest mistake investors make is to believe that what happened in the recent past is likely to persist.",
            "He who lives by the crystal ball will eat shattered glass.",
            "Don't get hung up on your views about how things should be.",
            "Pain plus reflection equals progress.",
            "The economy is like a machine.",
            "Diversifying well is the most important thing you need to do in order to invest well."
        ]
        
        # Set Dalio-inspired strategy preferences
        self.set_strategy_preferences({
            'macro_analysis': 0.95,        # Very high focus on macro trends
            'systematic_trading': 0.9,      # Strong emphasis on systematic approach
            'risk_parity': 0.9,            # High focus on risk-balanced portfolio
            'diversification': 0.95,        # Very high emphasis on diversification
            'trend_following': 0.7,         # Moderate trend following
            'correlation_trading': 0.8,     # Strong interest in correlations
            'fundamental_analysis': 0.8,    # Strong emphasis on fundamentals
            'meme_coin_interest': 0.2,     # Low interest in meme coins
            'alt_coin_interest': 0.6       # Moderate interest in established alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'long-term',
            'preferred_condition': 'all-weather',
            'market_approach': 'systematic-global',
            'volatility_view': 'must-be-balanced',
            'crypto_view': 'emerging-asset-class',
            'meme_coin_view': 'too-speculative',
            'innovation_view': 'part-of-evolution'
        })
        
        # Technical parameters
        self.correlation_window = 90       # Window for correlation analysis
        self.volatility_window = 30        # Window for volatility calculation
        self.trend_window = 200           # Window for long-term trend
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market data using global macro perspective."""
        analysis = {}
        
        try:
            if market_data.empty or len(market_data) < self.correlation_window:
                return {'error': 'Insufficient data for analysis'}
            
            # Calculate basic metrics
            close_prices = market_data['close']
            volumes = market_data['volume']
            
            # Calculate volatility
            returns = close_prices.pct_change()
            volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(365)
            
            # Calculate trends
            sma_long = close_prices.rolling(window=self.trend_window).mean()
            sma_short = close_prices.rolling(window=20).mean()
            
            # Calculate momentum
            rsi = ta.momentum.RSIIndicator(close_prices).rsi()
            macd = ta.trend.MACD(close_prices)
            
            # Current market conditions
            current_price = close_prices.iloc[-1]
            current_vol = volatility.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Trend analysis
            is_uptrend = current_price > sma_long.iloc[-1]
            trend_strength = (current_price - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            analysis = {
                'current_price': current_price,
                'volatility': current_vol,
                'rsi': current_rsi,
                'is_uptrend': is_uptrend,
                'trend_strength': trend_strength,
                'macd_hist': macd.macd_diff().iloc[-1],
                'volume_trend': volumes.iloc[-1] / volumes.rolling(window=20).mean().iloc[-1]
            }
            
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """Generate trading signal based on macro analysis."""
        if 'error' in analysis:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f"Analysis error: {analysis['error']}"}
        
        # Initialize signal components
        action = 'HOLD'
        confidence = 0.5
        reason = []
        
        # Analyze trend and momentum
        if analysis['is_uptrend']:
            if analysis['trend_strength'] > 0.05:
                if analysis['rsi'] < 70:  # Not overbought
                    action = 'BUY'
                    confidence = 0.7
                    reason.append("Strong uptrend with reasonable RSI")
                else:
                    action = 'HOLD'
                    reason.append("Uptrend but overbought")
        else:
            if analysis['trend_strength'] < -0.05:
                if analysis['rsi'] > 30:  # Not oversold
                    action = 'SELL'
                    confidence = 0.7
                    reason.append("Strong downtrend with reasonable RSI")
                else:
                    action = 'HOLD'
                    reason.append("Downtrend but oversold")
        
        # Consider volatility
        if analysis['volatility'] > 0.8:  # High volatility
            confidence = max(0.5, confidence - 0.1)
            reason.append("High volatility - reducing position size")
        elif analysis['volatility'] < 0.2:  # Low volatility
            confidence = min(0.9, confidence + 0.1)
            reason.append("Low volatility - increasing confidence")
        
        # Volume confirmation
        if analysis['volume_trend'] > 1.5:
            confidence = min(0.9, confidence + 0.1)
            reason.append("Strong volume supporting move")
        
        # MACD confirmation
        if analysis['macd_hist'] > 0 and action == 'BUY':
            confidence = min(0.9, confidence + 0.1)
            reason.append("MACD confirms bullish trend")
        elif analysis['macd_hist'] < 0 and action == 'SELL':
            confidence = min(0.9, confidence + 0.1)
            reason.append("MACD confirms bearish trend")
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': " | ".join(reason) or "No clear macro opportunity"
        }
    
    def get_personality_traits(self) -> Dict[str, str]:
        """Get personality traits of the agent."""
        traits = super().get_personality_traits()
        
        # Add Dalio-specific traits
        traits['personality'] = "Macro"
        traits['investment_philosophy'] = "Systematic global macro with risk parity"
        traits['famous_quote'] = np.random.choice(self.quotes)
        traits['risk_approach'] = "Balance risks across different market environments"
        traits['time_horizon'] = "Long-term, focused on secular trends"
        traits['meme_coin_approach'] = "Avoid unless part of broader market dynamic"
        
        return traits 