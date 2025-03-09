from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Literal, Any
import pandas as pd
from datetime import datetime

from trader.data.wallet import Wallet
from trader.data.market_data import MarketDataFetcher

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
    """
    Base class for all trading agents.
    Provides common functionality and defines the interface for agent-specific implementations.
    """
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
        self.strategy_preferences: Dict[str, float] = {}
        self.market_beliefs: Dict[str, str] = {}
        self.wallet = Wallet(initial_balance_usdt=20.0)
        self.data_fetcher = MarketDataFetcher()
        self.signals_history: List[Dict] = []
        
        # Action thresholds for different signal types
        self.action_thresholds = {
            'STRONG_BUY': 0.9,    # Very high confidence threshold
            'BUY': 0.7,           # Standard buy threshold
            'SCALE_IN': 0.6,      # Gradual position increase threshold
            'WATCH': 0.4,         # Monitoring threshold
            'HOLD': 0.0,          # Default action
            'SCALE_OUT': 0.6,     # Gradual position decrease threshold
            'SELL': 0.7,          # Standard sell threshold
            'STRONG_SELL': 0.9    # Very high confidence threshold
        }
        
    def set_strategy_preferences(self, preferences: Dict[str, float]) -> None:
        """
        Set the agent's strategy preferences.
        
        Args:
            preferences: Dictionary of strategy preferences (strategy -> weight)
        """
        self.strategy_preferences = preferences
        
    def update_market_beliefs(self, beliefs: Dict[str, str]) -> None:
        """
        Update the agent's market beliefs.
        
        Args:
            beliefs: Dictionary of market beliefs (belief -> value)
        """
        self.market_beliefs = {**self.market_beliefs, **beliefs}
        
    @abstractmethod
    def analyze_market(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Trading pair symbol
            data: DataFrame with market data
            
        Returns:
            Dictionary with analysis results
        """
        pass
        
    def generate_signal(self, symbol: str, confidence: float, reason: str = "") -> Dict:
        """
        Generate a trading signal based on confidence level.
        
        Args:
            symbol: Trading pair symbol
            confidence: Confidence level (-1.0 to 1.0)
            reason: Reason for the signal
            
        Returns:
            Signal dictionary
        """
        # Determine action based on confidence
        action = 'HOLD'
        
        if confidence >= self.action_thresholds['STRONG_BUY']:
            action = 'STRONG_BUY'
        elif confidence >= self.action_thresholds['BUY']:
            action = 'BUY'
        elif confidence >= self.action_thresholds['SCALE_IN']:
            action = 'SCALE_IN'
        elif confidence <= -self.action_thresholds['STRONG_SELL']:
            action = 'STRONG_SELL'
        elif confidence <= -self.action_thresholds['SELL']:
            action = 'SELL'
        elif confidence <= -self.action_thresholds['SCALE_OUT']:
            action = 'SCALE_OUT'
        elif abs(confidence) < self.action_thresholds['WATCH']:
            action = 'HOLD'  # Low confidence, just hold
        elif confidence > 0:
            action = 'WATCH'  # Positive but not strong enough to buy
        else:
            action = 'WATCH'  # Negative but not strong enough to sell
            
        # Get current market data for volatility-based position sizing
        try:
            data = self.data_fetcher.get_market_data(symbol, self.timeframe, limit=20)
            if not data.empty:
                # Calculate volatility (using ATR or standard deviation)
                if len(data) >= 14:
                    # Calculate 14-period volatility as percentage of price
                    volatility = data['close'].pct_change().rolling(14).std().iloc[-1]
                    # Adjust for missing values
                    volatility = volatility if not pd.isna(volatility) else 0.02  # Default 2% volatility
                else:
                    volatility = 0.02  # Default 2% volatility
            else:
                volatility = 0.02  # Default 2% volatility
        except Exception:
            volatility = 0.02  # Default 2% volatility
            
        # Inverse relationship between volatility and position size
        # Higher volatility = smaller position size
        volatility_factor = max(0.2, min(1.0, 0.05 / max(0.005, volatility)))
            
        # Calculate position size based on confidence, risk tolerance, and volatility
        current_price = self.data_fetcher.fetch_current_price(symbol)
        available_funds = self.wallet.balance_usdt
        
        # Adjust position size based on confidence, risk tolerance, and volatility
        position_size_pct = abs(confidence) * self.risk_tolerance * volatility_factor
        
        # Calculate amount in USDT
        amount_usdt = available_funds * position_size_pct
        
        # Ensure minimum trade amount
        min_trade_amount = 3.0  # $3 minimum
        if amount_usdt < min_trade_amount:
            amount_usdt = min_trade_amount if available_funds >= min_trade_amount else 0
            
        # For SCALE_IN actions, reduce the position size
        if action == 'SCALE_IN':
            amount_usdt = amount_usdt * 0.5  # 50% of normal position size
            
        # Calculate amount in crypto
        amount_crypto = amount_usdt / current_price if current_price > 0 else 0
        
        # For sell actions, check current holdings
        if action in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
            current_holdings = self.wallet.get_position_size(symbol)
            if current_holdings > 0:
                # Adjust sell amount based on action type
                if action == 'STRONG_SELL':
                    amount_crypto = current_holdings  # Sell all
                elif action == 'SELL':
                    amount_crypto = current_holdings * 0.75  # Sell 75%
                elif action == 'SCALE_OUT':
                    amount_crypto = current_holdings * 0.25  # Sell 25%
            else:
                # No holdings to sell
                action = 'HOLD'
                amount_crypto = 0
                
        # Create signal
        signal = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'amount_usdt': amount_usdt if action in ['BUY', 'STRONG_BUY', 'SCALE_IN'] else 0,
            'amount_crypto': amount_crypto if action in ['SELL', 'STRONG_SELL', 'SCALE_OUT'] else 0,
            'price': current_price,
            'reason': reason,
            'volatility': volatility if 'volatility' in locals() else 0.02,
            'volatility_factor': volatility_factor if 'volatility_factor' in locals() else 1.0
        }
        
        # Record signal
        self.signals_history.append(signal)
        
        return signal
        
    def execute_signal(self, signal: Dict) -> bool:
        """
        Execute a trading signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if the trade was successful, False otherwise
        """
        symbol = signal.get('symbol', '')
        current_price = signal.get('price', 0.0)
        
        if not symbol or current_price <= 0:
            return False
            
        return self.wallet.execute_trade(symbol, signal, current_price)
        
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the agent.
        
        Returns:
            Dictionary of performance metrics
        """
        # Get current prices for all holdings
        symbols = list(self.wallet.holdings.keys())
        prices = self.data_fetcher.fetch_multiple_prices(symbols)
        
        # Get wallet performance metrics
        metrics = self.wallet.get_performance_metrics(prices)
        
        # Add agent-specific metrics
        metrics.update({
            'name': self.name,
            'personality': self.personality,
            'risk_tolerance': self.risk_tolerance,
            'timeframe': self.timeframe,
            'signals_count': len(self.signals_history),
            'strategy_preferences': self.strategy_preferences,
            'market_beliefs': self.market_beliefs
        })
        
        return metrics
        
    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.wallet.reset()
        self.signals_history = [] 