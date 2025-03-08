import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class SwingTrader(BaseAgent):
    def __init__(self, name: str = "Jesse Livermore AI", timeframe: str = '4h'):
        """
        Initialize the Swing Trader agent inspired by Jesse Livermore.
        Known for aggressive trading style and excellent market timing.
        """
        super().__init__(
            name=name,
            personality="Swing Trader - Aggressive, momentum-focused, market timing expert, high conviction",
            risk_tolerance=0.9,  # Very high risk tolerance
            timeframe=timeframe
        )
        
        # Jesse Livermore's famous quotes
        self.quotes = [
            "The market is never wrong – opinions often are.",
            "Don't take action with a trade until the market confirms your opinion.",
            "Markets are never wrong – opinions often are.",
            "The game of speculation is the most uniformly fascinating game in the world.",
            "There is nothing new in Wall Street. There can't be because speculation is as old as the hills.",
            "The human side of every person is the greatest enemy of the average investor or speculator."
        ]
        
        # Set Livermore-inspired strategy preferences
        self.set_strategy_preferences({
            'swing_trading': 0.95,         # Very high focus on swing trading
            'momentum_trading': 0.9,       # Strong emphasis on momentum
            'trend_following': 0.8,        # Strong trend following
            'breakout_trading': 0.9,       # High focus on breakouts
            'market_timing': 0.95,         # Very high emphasis on timing
            'position_sizing': 0.9,        # Strong focus on position sizing
            'technical_analysis': 0.9,     # Strong technical analysis
            'meme_coin_interest': 0.7,     # High interest in volatile meme coins
            'alt_coin_interest': 0.8       # High interest in alt coins
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'short-to-medium',
            'preferred_condition': 'trending-volatile',
            'market_approach': 'technical-momentum',
            'volatility_view': 'opportunity-for-profit',
            'crypto_view': 'trading-vehicle',
            'meme_coin_view': 'tradable-with-momentum',
            'innovation_view': 'focus-on-price-action'
        })
        
        # Technical parameters
        self.pivot_window = 20            # Window for pivot point calculation
        self.momentum_window = 14         # Window for momentum indicators
        self.breakout_threshold = 2.0     # Standard deviations for breakout
        
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market data for swing trading opportunities."""
        analysis = {}
        
        try:
            if market_data.empty or len(market_data) < self.pivot_window:
                return {'error': 'Insufficient data for analysis'}
            
            # Calculate basic metrics
            close_prices = market_data['close']
            high_prices = market_data['high']
            low_prices = market_data['low']
            volumes = market_data['volume']
            
            # Calculate momentum indicators
            rsi = ta.momentum.RSIIndicator(close_prices).rsi()
            stoch = ta.momentum.StochasticOscillator(high_prices, low_prices, close_prices)
            macd = ta.trend.MACD(close_prices)
            
            # Calculate volatility and momentum
            atr = ta.volatility.AverageTrueRange(high_prices, low_prices, close_prices).average_true_range()
            
            # Current market conditions
            current_price = close_prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_stoch_k = stoch.stoch().iloc[-1]
            current_stoch_d = stoch.stoch_signal().iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Identify potential pivot points
            rolling_high = high_prices.rolling(window=self.pivot_window).max()
            rolling_low = low_prices.rolling(window=self.pivot_window).min()
            
            # Check for breakouts
            upper_band = rolling_high.shift(1)
            lower_band = rolling_low.shift(1)
            
            analysis = {
                'current_price': current_price,
                'rsi': current_rsi,
                'stoch_k': current_stoch_k,
                'stoch_d': current_stoch_d,
                'macd_hist': macd.macd_diff().iloc[-1],
                'atr': current_atr,
                'above_upper_band': current_price > upper_band.iloc[-1],
                'below_lower_band': current_price < lower_band.iloc[-1],
                'volume_surge': volumes.iloc[-1] > volumes.rolling(window=20).mean().iloc[-1] * 1.5
            }
            
        except Exception as e:
            analysis['error'] = str(e)
            
        return analysis
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """Generate trading signal based on swing trading analysis."""
        if 'error' in analysis:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f"Analysis error: {analysis['error']}"}
        
        # Initialize signal components
        action = 'HOLD'
        confidence = 0.5
        reason = []
        
        # Check for breakout conditions
        if analysis['above_upper_band']:
            if analysis['volume_surge']:
                action = 'STRONG_BUY'
                confidence = 0.9
                reason.append("Strong breakout with volume confirmation")
            else:
                action = 'BUY'
                confidence = 0.7
                reason.append("Breakout detected")
        elif analysis['below_lower_band']:
            if analysis['volume_surge']:
                action = 'STRONG_SELL'
                confidence = 0.9
                reason.append("Strong breakdown with volume confirmation")
            else:
                action = 'SELL'
                confidence = 0.7
                reason.append("Breakdown detected")
        
        # Consider momentum indicators
        if analysis['rsi'] > 70:
            if action != 'STRONG_SELL':
                action = 'SELL'
                confidence = 0.8
                reason.append("Overbought conditions")
        elif analysis['rsi'] < 30:
            if action != 'STRONG_BUY':
                action = 'BUY'
                confidence = 0.8
                reason.append("Oversold conditions")
        
        # Stochastic confirmation
        if analysis['stoch_k'] > analysis['stoch_d'] and action in ['BUY', 'STRONG_BUY']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("Stochastic confirms upward momentum")
        elif analysis['stoch_k'] < analysis['stoch_d'] and action in ['SELL', 'STRONG_SELL']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("Stochastic confirms downward momentum")
        
        # MACD confirmation
        if analysis['macd_hist'] > 0 and action in ['BUY', 'STRONG_BUY']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("MACD confirms bullish momentum")
        elif analysis['macd_hist'] < 0 and action in ['SELL', 'STRONG_SELL']:
            confidence = min(0.95, confidence + 0.1)
            reason.append("MACD confirms bearish momentum")
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': " | ".join(reason) or "No clear swing trading opportunity"
        }
    
    def get_personality_traits(self) -> Dict[str, str]:
        """Get personality traits of the agent."""
        traits = super().get_personality_traits()
        
        # Add Livermore-specific traits
        traits['personality'] = "Swing"
        traits['investment_philosophy'] = "Aggressive swing trading with perfect market timing"
        traits['famous_quote'] = np.random.choice(self.quotes)
        traits['risk_approach'] = "High risk, high conviction trades with strict rules"
        traits['time_horizon'] = "Short to medium-term, focused on momentum"
        traits['meme_coin_approach'] = "Trade any coin showing strong momentum"
        
        return traits 