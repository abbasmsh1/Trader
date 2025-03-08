from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Literal
import pandas as pd
from data.market_news import MarketNewsFetcher

# Define trading actions as constants
TradingAction = Literal[
    'STRONG_BUY',    # Very confident buy signal
    'BUY',           # Standard buy signal
    'SCALE_IN',      # Gradually increase position
    'WATCH',         # Monitor for opportunities
    'HOLD',          # No action needed
    'SCALE_OUT',     # Gradually decrease position
    'SELL',          # Standard sell signal
    'STRONG_SELL'    # Very confident sell signal
]

class BaseAgent(ABC):
    def __init__(self, name: str, personality: str, risk_tolerance: float = 0.5, timeframe: str = '1h'):
        """
        Initialize the base trading agent.
        
        Args:
            name (str): Name of the trading agent
            personality (str): Trading personality description
            risk_tolerance (float): Risk tolerance level between 0 and 1
            timeframe (str): Trading timeframe (e.g., '1h', '4h', '1d')
        """
        self.name = name
        self.personality = personality
        self.risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        self.timeframe = timeframe
        self.portfolio: Dict[str, float] = {}
        self.trading_history: List[Dict] = []
        self.strategy_preferences: Dict[str, float] = {}
        self.market_beliefs: Dict[str, str] = {}
        self.news_fetcher = MarketNewsFetcher()
        
        # Action thresholds for different signal types
        self.action_thresholds = {
            'STRONG_BUY': 0.9,    # Very high confidence threshold
            'BUY': 0.7,           # Standard buy threshold
            'SCALE_IN': 0.6,      # Gradual position increase threshold
            'WATCH': 0.4,         # Monitoring threshold
            'HOLD': 0.0,          # Default action
            'SCALE_OUT': 0.6,     # Gradual position decrease threshold
            'SELL': 0.7,          # Standard sell threshold
            'STRONG_SELL': 0.9    # Very high confidence for selling
        }
        
    @abstractmethod
    def analyze_market(self, market_data: pd.DataFrame) -> Dict:
        """
        Analyze market data and generate insights.
        
        Args:
            market_data (pd.DataFrame): Historical market data
            
        Returns:
            Dict: Analysis results and insights
        """
        pass
    
    @abstractmethod
    def generate_signal(self, analysis: Dict) -> Dict:
        """
        Generate trading signals based on analysis.
        
        Args:
            analysis (Dict): Market analysis results
            
        Returns:
            Dict: Trading signal with action, price, and confidence
        """
        pass
    
    def determine_action(self, confidence: float, trend_direction: int) -> TradingAction:
        """
        Determine the appropriate trading action based on confidence and trend.
        
        Args:
            confidence (float): Signal confidence level (0 to 1)
            trend_direction (int): 1 for uptrend, -1 for downtrend
            
        Returns:
            TradingAction: The determined trading action
        """
        if trend_direction > 0:  # Uptrend
            if confidence >= self.action_thresholds['STRONG_BUY']:
                return 'STRONG_BUY'
            elif confidence >= self.action_thresholds['BUY']:
                return 'BUY'
            elif confidence >= self.action_thresholds['SCALE_IN']:
                return 'SCALE_IN'
            elif confidence >= self.action_thresholds['WATCH']:
                return 'WATCH'
            else:
                return 'HOLD'
        elif trend_direction < 0:  # Downtrend
            if confidence >= self.action_thresholds['STRONG_SELL']:
                return 'STRONG_SELL'
            elif confidence >= self.action_thresholds['SELL']:
                return 'SELL'
            elif confidence >= self.action_thresholds['SCALE_OUT']:
                return 'SCALE_OUT'
            elif confidence >= self.action_thresholds['WATCH']:
                return 'WATCH'
            else:
                return 'HOLD'
        else:  # No clear trend
            return 'WATCH'
    
    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """
        Calculate the position size based on risk tolerance, signal confidence, and action type.
        
        Args:
            signal (Dict): Trading signal
            available_capital (float): Available capital for trading
            
        Returns:
            float: Recommended position size
        """
        confidence = signal.get('confidence', 0.5)
        action = signal.get('action', 'HOLD')
        
        # Adjust position size based on action type
        action_multipliers = {
            'STRONG_BUY': 1.0,
            'BUY': 0.8,
            'SCALE_IN': 0.4,
            'WATCH': 0.0,
            'HOLD': 0.0,
            'SCALE_OUT': 0.4,
            'SELL': 0.8,
            'STRONG_SELL': 1.0
        }
        
        action_multiplier = action_multipliers.get(action, 0.0)
        risk_adjusted_size = available_capital * self.risk_tolerance * confidence * action_multiplier
        return risk_adjusted_size
    
    def update_portfolio(self, trade: Dict):
        """
        Update the agent's portfolio after a trade.
        
        Args:
            trade (Dict): Trade details including asset, amount, and price
        """
        asset = trade['asset']
        amount = trade['amount']
        self.portfolio[asset] = self.portfolio.get(asset, 0) + amount
        self.trading_history.append(trade)
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate agent's performance metrics.
        
        Returns:
            Dict: Performance metrics including returns, win rate, etc.
        """
        if not self.trading_history:
            return {'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0}
        
        total_trades = len(self.trading_history)
        profitable_trades = sum(1 for trade in self.trading_history if trade.get('profit', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        total_return = sum(trade.get('profit', 0) for trade in self.trading_history)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'personality': self.personality,
            'strategy': self.strategy_preferences,
            'market_view': self.market_beliefs
        }
    
    def set_strategy_preferences(self, preferences: Dict[str, float]):
        """Set agent's strategy preferences."""
        self.strategy_preferences = preferences
    
    def update_market_beliefs(self, beliefs: Dict[str, str]):
        """Update agent's market beliefs."""
        self.market_beliefs = beliefs
    
    def get_personality_traits(self) -> Dict[str, str]:
        """Get agent's personality traits and characteristics."""
        return {
            'name': self.name,
            'personality': self.personality,
            'risk_tolerance': self.risk_tolerance,
            'preferred_timeframe': self.timeframe,
            'strategy_preferences': self.strategy_preferences,
            'market_beliefs': self.market_beliefs
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.personality}) - Risk: {self.risk_tolerance:.2f}, Timeframe: {self.timeframe}"
    
    def analyze_news_sentiment(self, symbol: str) -> Dict:
        """
        Analyze market news sentiment.
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Dict: News sentiment analysis results
        """
        return self.news_fetcher.get_market_sentiment(symbol, self.timeframe)
        
    def adjust_confidence(self, base_confidence: float, sentiment: Dict) -> float:
        """
        Adjust signal confidence based on news sentiment.
        
        Args:
            base_confidence (float): Base confidence from technical analysis
            sentiment (Dict): News sentiment analysis results
            
        Returns:
            float: Adjusted confidence score
        """
        if sentiment['article_count'] == 0:
            return base_confidence
            
        # Calculate sentiment impact (0.0 to 0.3)
        sentiment_impact = min(0.3, sentiment['confidence'] * abs(sentiment['sentiment_score']))
        
        # Adjust confidence based on sentiment alignment
        if sentiment['sentiment_score'] > 0 and base_confidence > 0.5:
            # Positive sentiment reinforces bullish signal
            return min(1.0, base_confidence + sentiment_impact)
        elif sentiment['sentiment_score'] < 0 and base_confidence < 0.5:
            # Negative sentiment reinforces bearish signal
            return max(0.0, base_confidence - sentiment_impact)
        else:
            # Sentiment contradicts signal, reduce confidence
            return base_confidence * (1.0 - sentiment_impact)
            
    def get_news_commentary(self, sentiment: Dict) -> str:
        """
        Generate commentary about market news sentiment.
        
        Args:
            sentiment (Dict): News sentiment analysis results
            
        Returns:
            str: News-based market commentary
        """
        if sentiment['article_count'] == 0:
            return "No recent news available."
            
        sentiment_str = sentiment['overall_sentiment'].capitalize()
        confidence_str = "High" if sentiment['confidence'] > 0.7 else "Moderate" if sentiment['confidence'] > 0.4 else "Low"
        
        return f"Market news sentiment: {sentiment_str} ({confidence_str} confidence). " + \
               f"Based on {sentiment['article_count']} recent articles " + \
               f"({sentiment['positive_ratio']*100:.0f}% positive, " + \
               f"{sentiment['negative_ratio']*100:.0f}% negative)." 