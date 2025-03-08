import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class ContrarianTrader(BaseAgent):
    def __init__(self, name: str = "Michael Burry AI", timeframe: str = '1d'):
        """
        Initialize the Contrarian Trader agent inspired by Michael Burry.
        Known for identifying market bubbles and making contrarian bets.
        """
        super().__init__(
            name=name,
            personality="Contrarian - Data-driven, bubble spotter, against-the-crowd, deep research",
            risk_tolerance=0.7,  # High risk tolerance for contrarian bets
            timeframe=timeframe
        )
        
        # Michael Burry's famous quotes
        self.quotes = [
            "I invest in stocks that are undervalued and shorted.",
            "The entire derivatives market is a massive time bomb.",
            "I started getting a lot of attention when I started shorting subprime.",
            "Markets are driven by humans and human nature never changes.",
            "I focus on what everyone else has missed.",
            "I am 100% focused on identifying frauds and bubbles."
        ]
        
        # Set Burry-inspired strategy preferences
        self.set_strategy_preferences({
            'contrarian_focus': 0.95,      # Very high focus on contrarian plays
            'bubble_detection': 0.9,       # Strong emphasis on identifying bubbles
            'value_investing': 0.8,        # Significant value component
            'trend_following': 0.2,        # Low interest in following trends
            'momentum_trading': 0.3,       # Limited interest in momentum
            'short_selling': 0.9,          # High interest in shorting overvalued assets
            'fundamental_analysis': 0.9,    # Strong emphasis on fundamentals
            'meme_coin_interest': 0.1,     # Very low interest in meme coins
            'alt_coin_interest': 0.4       # Moderate interest in established alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'medium-to-long-term',
            'preferred_condition': 'overvalued-markets',
            'market_approach': 'contrarian-analytical',
            'volatility_view': 'opportunity-for-profit',
            'crypto_view': 'bubble-prone-asset-class',
            'meme_coin_view': 'likely-ponzi-schemes',
            'innovation_view': 'skeptical-but-analytical'
        })
        
        # Technical parameters
        self.bubble_threshold = 2.0        # Price deviation from mean
        self.volume_surge_threshold = 3.0  # Volume increase threshold
        self.correlation_window = 90       # Window for correlation analysis
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market data for bubble conditions and contrarian opportunities."""
        analysis = {}
        
        try:
            if market_data.empty or len(market_data) < self.correlation_window:
                return {'error': 'Insufficient data for analysis'}
            
            # Calculate basic metrics
            close_prices = market_data['close']
            volumes = market_data['volume']
            
            # Identify potential bubble conditions
            price_mean = close_prices.rolling(window=self.correlation_window).mean()
            price_std = close_prices.rolling(window=self.correlation_window).std()
            z_score = (close_prices - price_mean) / price_std
            
            # Calculate momentum and trend indicators
            rsi = ta.momentum.RSIIndicator(close_prices).rsi()
            macd = ta.trend.MACD(close_prices)
            
            # Current market conditions
            current_z_score = z_score.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_price = close_prices.iloc[-1]
            
            # Identify bubble conditions
            is_bubble = abs(current_z_score) > self.bubble_threshold
            is_overbought = current_rsi > 70
            
            # Volume analysis
            volume_sma = volumes.rolling(window=20).mean()
            volume_surge = volumes.iloc[-1] > volume_sma.iloc[-1] * self.volume_surge_threshold
            
            analysis = {
                'is_bubble': is_bubble,
                'z_score': current_z_score,
                'is_overbought': is_overbought,
                'rsi': current_rsi,
                'volume_surge': volume_surge,
                'current_price': current_price,
                'macd_hist': macd.macd_diff().iloc[-1]
            }
            
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """Generate trading signal based on contrarian analysis."""
        if 'error' in analysis:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f"Analysis error: {analysis['error']}"}
        
        # Initialize signal components
        action = 'HOLD'
        confidence = 0.5
        reason = []
        
        # Check for bubble conditions
        if analysis['is_bubble']:
            if analysis['z_score'] > self.bubble_threshold:
                action = 'STRONG_SELL'
                confidence = min(0.9, 0.6 + abs(analysis['z_score']) / 10)
                reason.append("Potential bubble detected")
            elif analysis['z_score'] < -self.bubble_threshold:
                action = 'STRONG_BUY'
                confidence = min(0.9, 0.6 + abs(analysis['z_score']) / 10)
                reason.append("Asset potentially oversold")
        
        # Consider RSI for contrarian plays
        if analysis['is_overbought'] and analysis['rsi'] > 80:
            if action != 'STRONG_SELL':
                action = 'SELL'
                confidence = 0.7
                reason.append("Overbought conditions")
        elif analysis['rsi'] < 30:
            if action != 'STRONG_BUY':
                action = 'BUY'
                confidence = 0.7
                reason.append("Oversold conditions")
        
        # Volume confirmation
        if analysis['volume_surge']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("High volume supporting the move")
        
        # MACD divergence
        if analysis['macd_hist'] < 0 and action in ['STRONG_SELL', 'SELL']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("MACD confirms bearish momentum")
        elif analysis['macd_hist'] > 0 and action in ['STRONG_BUY', 'BUY']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("MACD confirms bullish momentum")
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': " | ".join(reason) or "No clear contrarian opportunity"
        }
    
    def get_personality_traits(self) -> Dict[str, str]:
        """Get personality traits of the agent."""
        traits = super().get_personality_traits()
        
        # Add Burry-specific traits
        traits['personality'] = "Contrarian"
        traits['investment_philosophy'] = "Identify and profit from market inefficiencies and bubbles"
        traits['famous_quote'] = np.random.choice(self.quotes)
        traits['risk_approach'] = "High conviction contrarian bets with thorough research"
        traits['time_horizon'] = "Medium to long-term, waiting for thesis to play out"
        traits['meme_coin_approach'] = "Highly skeptical, looks for signs of bubble"
        
        return traits 