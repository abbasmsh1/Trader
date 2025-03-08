from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional
import pandas as pd

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
    
    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """
        Calculate the position size based on risk tolerance and signal confidence.
        
        Args:
            signal (Dict): Trading signal
            available_capital (float): Available capital for trading
            
        Returns:
            float: Recommended position size
        """
        confidence = signal.get('confidence', 0.5)
        risk_adjusted_size = available_capital * self.risk_tolerance * confidence
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