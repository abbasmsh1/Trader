import pandas as pd
import numpy as np
from typing import Dict
import ta
from .base_agent import BaseAgent

class TrendFollower(BaseAgent):
    def __init__(self, name: str = "Technical Trader", risk_tolerance: float = 0.6, timeframe: str = '4h'):
        """
        Initialize the Trend Follower agent.
        This agent follows market trends using moving averages and momentum indicators.
        Inspired by systematic traders like Paul Tudor Jones and trend-following CTAs.
        """
        super().__init__(
            name=name,
            personality="Technical Trader - Systematic, trend-focused, data-driven, disciplined, unemotional",
            risk_tolerance=risk_tolerance,
            timeframe=timeframe
        )
        
        # Technical trader quotes
        self.quotes = [
            "The trend is your friend until it bends.",
            "Cut your losses short and let your winners run.",
            "Don't fight the tape.",
            "Plan your trade and trade your plan.",
            "The market can remain irrational longer than you can remain solvent.",
            "Risk management is more important than prediction.",
            "Price action never lies."
        ]
        
        # Technical parameters
        self.short_window = 20
        self.long_window = 50
        self.rsi_period = 14
        self.rsi_overbought = 70.0  # Ensure float
        self.rsi_oversold = 30.0    # Ensure float
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_window = 20
        self.bollinger_std = 2.0
        
        # Set Technical Trader strategy preferences
        self.set_strategy_preferences({
            'trend_following': 0.95,       # Very high focus on trends
            'momentum_trading': 0.8,       # Strong interest in momentum
            'mean_reversion': 0.6,         # Moderate interest in mean reversion
            'breakout_trading': 0.85,      # High interest in breakouts
            'volatility_trading': 0.7,     # Significant interest in volatility
            'technical_indicators': 0.9,   # Very high reliance on indicators
            'chart_patterns': 0.8,         # Strong interest in chart patterns
            'meme_coin_interest': 0.5,     # Moderate interest in meme coins (if they show good technicals)
            'alt_coin_interest': 0.7       # Significant interest in alt coins with good technical setups
        })
        
        # Initial market beliefs
        self.update_market_beliefs({
            'time_horizon': 'medium-term',
            'preferred_condition': 'trending-markets',
            'market_approach': 'technical-analysis-driven',
            'volatility_view': 'opportunity-with-risk-management',
            'crypto_view': 'tradable-asset-class',
            'meme_coin_view': 'tradable-if-technical-setup-is-good',
            'fundamental_value': 'irrelevant-compared-to-price-action'
        })
        
    def get_personality_traits(self) -> Dict[str, str]:
        """Get personality traits of the agent."""
        traits = super().get_personality_traits()
        
        # Add Technical Trader specific traits
        traits['personality'] = "Technical"
        traits['investment_philosophy'] = "Follow price action and technical indicators, ignore fundamentals"
        traits['famous_quote'] = np.random.choice(self.quotes)
        traits['risk_approach'] = "Disciplined risk management with strict stop losses"
        traits['time_horizon'] = "Medium-term, follows trends until they reverse"
        traits['meme_coin_approach'] = "Agnostic to coin type, trades based on technical patterns only"
        traits['trading_style'] = "Systematic, rule-based, emotionless"
        
        return traits
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data using trend-following indicators and news sentiment.
        
        Args:
            market_data (pd.DataFrame): Historical market data with OHLCV
            
        Returns:
            Dict: Analysis results including trend direction and strength
        """
        df = market_data.copy()
        
        # Calculate moving averages
        df['SMA_short'] = ta.trend.sma_indicator(df['close'], self.short_window)
        df['SMA_long'] = ta.trend.sma_indicator(df['close'], self.long_window)
        
        # Calculate RSI
        df['RSI'] = ta.momentum.rsi(df['close'], self.rsi_period)
        
        # Calculate MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Ensure numeric values and handle NaN
        sma_short = float(latest['SMA_short']) if pd.notnull(latest['SMA_short']) else 0.0
        sma_long = float(latest['SMA_long']) if pd.notnull(latest['SMA_long']) else 0.0
        rsi = float(latest['RSI']) if pd.notnull(latest['RSI']) else 50.0
        macd_current = float(latest['MACD']) if pd.notnull(latest['MACD']) else 0.0
        macd_signal = float(latest['MACD_signal']) if pd.notnull(latest['MACD_signal']) else 0.0
        macd_prev = float(prev['MACD']) if pd.notnull(prev['MACD']) else 0.0
        macd_signal_prev = float(prev['MACD_signal']) if pd.notnull(prev['MACD_signal']) else 0.0
        
        # Determine trend direction and strength
        trend_direction = 1 if sma_short > sma_long else -1
        trend_strength = abs(sma_short - sma_long) / float(latest['close'])
        
        # Check for MACD crossover
        macd_crossover = (
            (macd_current > macd_signal and macd_prev <= macd_signal_prev) or
            (macd_current < macd_signal and macd_prev >= macd_signal_prev)
        )
        
        # Get news sentiment analysis
        symbol = market_data.name if hasattr(market_data, 'name') else 'UNKNOWN'
        sentiment = self.analyze_news_sentiment(symbol)
        
        # Adjust trend strength based on news sentiment
        if sentiment['article_count'] > 0:
            # Strengthen or weaken trend based on sentiment alignment
            if (trend_direction > 0 and sentiment['sentiment_score'] > 0) or \
               (trend_direction < 0 and sentiment['sentiment_score'] < 0):
                trend_strength *= (1 + abs(sentiment['sentiment_score']) * 0.2)
            else:
                trend_strength *= (1 - abs(sentiment['sentiment_score']) * 0.1)
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': float(trend_strength),
            'rsi': float(rsi),
            'macd_crossover': bool(macd_crossover),
            'price': float(latest['close']),
            'sentiment': sentiment,
            'timestamp': latest.name
        }
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on trend analysis and news sentiment.
        
        Args:
            analysis (Dict): Market analysis results
            
        Returns:
            Dict: Trading signal with action, price, and confidence
        """
        signal = {
            'action': 'HOLD',
            'price': float(analysis['price']),
            'confidence': 0.0,
            'timestamp': analysis['timestamp']
        }
        
        # Calculate base confidence from trend strength
        base_confidence = min(float(analysis['trend_strength']) * 10, 1.0)
        
        # Adjust confidence based on news sentiment
        adjusted_confidence = self.adjust_confidence(base_confidence, analysis['sentiment'])
        
        # Strong uptrend with confirmation and positive news
        if (analysis['trend_direction'] > 0 and
            float(analysis['rsi']) < self.rsi_overbought - 10 and
            analysis['macd_crossover'] and
            base_confidence > 0.8 and
            analysis['sentiment']['sentiment_score'] > 0.2):
            
            signal['action'] = 'STRONG_BUY'
            signal['confidence'] = adjusted_confidence * 0.95
            
        # Regular uptrend entry
        elif (analysis['trend_direction'] > 0 and
              float(analysis['rsi']) < self.rsi_overbought and
              analysis['macd_crossover']):
            
            signal['action'] = 'BUY'
            signal['confidence'] = adjusted_confidence * 0.85
            
        # Building position in uptrend
        elif (analysis['trend_direction'] > 0 and
              float(analysis['rsi']) < self.rsi_overbought + 5):
            
            signal['action'] = 'SCALE_IN'
            signal['confidence'] = adjusted_confidence * 0.75
            
        # Strong downtrend with confirmation and negative news
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold + 10 and
              analysis['macd_crossover'] and
              base_confidence > 0.8 and
              analysis['sentiment']['sentiment_score'] < -0.2):
            
            signal['action'] = 'STRONG_SELL'
            signal['confidence'] = adjusted_confidence * 0.9
            
        # Regular downtrend exit
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold and
              analysis['macd_crossover']):
            
            signal['action'] = 'SELL'
            signal['confidence'] = adjusted_confidence * 0.8
            
        # Reducing position in downtrend
        elif (analysis['trend_direction'] < 0 and
              float(analysis['rsi']) > self.rsi_oversold - 5):
            
            signal['action'] = 'SCALE_OUT'
            signal['confidence'] = adjusted_confidence * 0.7
            
        # Monitoring conditions
        elif abs(analysis['trend_direction']) < 0.5:
            signal['action'] = 'WATCH'
            signal['confidence'] = adjusted_confidence * 0.6
        
        # Add technical analysis commentary with news sentiment
        signal['commentary'] = self._generate_technical_commentary(analysis, signal['action'])
        
        return signal
    
    def _generate_technical_commentary(self, analysis: Dict, action: str) -> str:
        """Generate technical analysis commentary based on the action and news."""
        base_comment = ""
        if action == 'STRONG_BUY':
            base_comment = "Strong bullish trend confirmed! Multiple indicators showing buy signals! 📈⬆️"
        elif action == 'BUY':
            base_comment = "Bullish trend developing with positive MACD crossover. 📈"
        elif action == 'SCALE_IN':
            base_comment = "Uptrend continues, adding to position on pullback. 📊"
        elif action == 'STRONG_SELL':
            base_comment = "Strong bearish trend confirmed! Multiple indicators showing sell signals! 📉⬇️"
        elif action == 'SELL':
            base_comment = "Bearish trend developing with negative MACD crossover. 📉"
        elif action == 'SCALE_OUT':
            base_comment = "Downtrend continues, reducing position on bounce. 📊"
        elif action == 'WATCH':
            base_comment = "Consolidation phase - waiting for clear trend direction. 🔍"
        else:
            base_comment = "Neutral trend - maintaining current position. ⚖️"
        
        # Add technical metrics
        tech_comment = (
            f"RSI: {analysis['rsi']:.1f} | " +
            f"Trend Strength: {analysis['trend_strength']:.2f} | " +
            f"Direction: {'Bullish' if analysis['trend_direction'] > 0 else 'Bearish'}"
        )
        
        # Add news sentiment commentary
        news_comment = self.get_news_commentary(analysis['sentiment'])
        
        return f"{base_comment}\n{tech_comment}\n{news_comment}" 