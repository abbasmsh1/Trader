"""
Trend Following Strategy Agent - Implements trend following trading strategy.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from services.market_data import MarketDataService
from utils.technical_indicators import calculate_macd, calculate_bollinger_bands, calculate_atr

class TrendFollowingStrategyAgent(BaseAgent):
    """
    Trend Following Strategy Agent that implements trend following logic.
    
    This agent:
    - Identifies market trends using multiple timeframes
    - Uses MACD, Bollinger Bands, and ATR for trend confirmation
    - Implements proper entry and exit logic
    - Manages position sizing based on trend strength
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the trend following strategy agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="strategy")
        
        # Strategy parameters
        self.trend_threshold = config.get("trend_threshold", 0.02)  # 2% trend threshold
        self.trend_confirmation_periods = config.get("trend_confirmation_periods", [20, 50, 200])
        self.entry_signal_threshold = config.get("entry_signal_threshold", 0.7)
        self.exit_signal_threshold = config.get("exit_signal_threshold", 0.3)
        
        # Technical indicator parameters
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2)
        
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2)
        
        # Risk management
        self.position_size = config.get("position_size", 0.1)  # 10% of portfolio
        self.stop_loss = config.get("stop_loss", 0.05)  # 5% stop loss
        self.take_profit = config.get("take_profit", 0.15)  # 15% take profit
        self.max_positions = config.get("max_positions", 3)
        
        # Market data service
        self.market_data_service = MarketDataService(
            api_key=config.get("api_key"),
            secret=config.get("secret")
        )
        
        # Initialize state
        self.current_positions = {}
        self.trend_states = {}
        self.last_analysis = {}
        
        self.logger.info(f"Trend Following Strategy Agent initialized with threshold {self.trend_threshold}")
    
    def _identify_trend(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Identify the current trend and its strength.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (trend_direction, trend_strength)
        """
        # Calculate moving averages for different timeframes
        ma_values = {}
        for period in self.trend_confirmation_periods:
            ma_values[period] = data['close'].rolling(window=period).mean()
        
        # Calculate trend direction (1 for uptrend, -1 for downtrend, 0 for sideways)
        trend_direction = 0
        trend_strength = 0
        
        # Check if all MAs are aligned
        if all(ma_values[period].iloc[-1] > ma_values[period].iloc[-2] 
               for period in self.trend_confirmation_periods):
            trend_direction = 1
        elif all(ma_values[period].iloc[-1] < ma_values[period].iloc[-2] 
                 for period in self.trend_confirmation_periods):
            trend_direction = -1
        
        # Calculate trend strength based on price distance from MAs
        if trend_direction != 0:
            price = data['close'].iloc[-1]
            ma_distances = [abs(price - ma.iloc[-1]) / ma.iloc[-1] 
                          for ma in ma_values.values()]
            trend_strength = np.mean(ma_distances)
        
        return trend_direction, trend_strength
    
    def _calculate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trading signals using technical indicators.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary of signal strengths
        """
        # Calculate MACD
        macd, signal, _ = calculate_macd(
            data['close'], 
            self.macd_fast, 
            self.macd_slow, 
            self.macd_signal
        )
        
        # Calculate Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(
            data['close'], 
            self.bb_period, 
            self.bb_std
        )
        
        # Calculate ATR
        atr = calculate_atr(
            data['high'], 
            data['low'], 
            data['close'], 
            self.atr_period
        )
        
        # Calculate signal strengths
        macd_signal = 1.0 if macd.iloc[-1] > signal.iloc[-1] else -1.0
        
        bb_signal = 0
        if data['close'].iloc[-1] < lower.iloc[-1]:
            bb_signal = 1.0
        elif data['close'].iloc[-1] > upper.iloc[-1]:
            bb_signal = -1.0
        
        # Calculate volatility signal
        volatility = atr.iloc[-1] / data['close'].iloc[-1]
        volatility_signal = 1.0 if volatility > self.trend_threshold else 0.0
        
        # Combine signals
        total_signal = (macd_signal + bb_signal + volatility_signal) / 3.0
        
        return {
            "macd": macd_signal,
            "bollinger": bb_signal,
            "volatility": volatility_signal,
            "total": total_signal
        }
    
    def _should_enter_position(self, 
                             symbol: str, 
                             trend_direction: float,
                             trend_strength: float,
                             signals: Dict[str, float], 
                             price: float) -> bool:
        """
        Determine if we should enter a position.
        
        Args:
            symbol: Trading symbol
            trend_direction: Current trend direction
            trend_strength: Current trend strength
            signals: Dictionary of trading signals
            price: Current price
            
        Returns:
            True if we should enter a position
        """
        # Check if we already have a position
        if symbol in self.current_positions:
            return False
        
        # Check if we've reached max positions
        if len(self.current_positions) >= self.max_positions:
            return False
        
        # Check trend direction and strength
        if trend_direction == 0 or trend_strength < self.trend_threshold:
            return False
        
        # Check signal strength
        if signals["total"] < self.entry_signal_threshold:
            return False
        
        # Check if trend direction matches signals
        if trend_direction * signals["total"] < 0:
            return False
        
        return True
    
    def _should_exit_position(self, 
                            symbol: str, 
                            trend_direction: float,
                            trend_strength: float,
                            signals: Dict[str, float], 
                            price: float) -> bool:
        """
        Determine if we should exit a position.
        
        Args:
            symbol: Trading symbol
            trend_direction: Current trend direction
            trend_strength: Current trend strength
            signals: Dictionary of trading signals
            price: Current price
            
        Returns:
            True if we should exit a position
        """
        # Check if we have a position
        if symbol not in self.current_positions:
            return False
        
        position = self.current_positions[symbol]
        entry_price = position["entry_price"]
        
        # Check stop loss
        if price <= entry_price * (1 - self.stop_loss):
            return True
        
        # Check take profit
        if price >= entry_price * (1 + self.take_profit):
            return True
        
        # Check trend reversal
        if trend_direction * position["direction"] < 0:
            return True
        
        # Check exit signals
        if signals["total"] < -self.exit_signal_threshold:
            return True
        
        # Check trend strength
        if trend_strength < self.trend_threshold:
            return True
        
        return False
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the trend following strategy.
        
        Args:
            data: Dictionary of market data
            
        Returns:
            Dictionary of trading decisions
        """
        decisions = {}
        
        for symbol, symbol_data in data.items():
            # Convert to DataFrame
            df = pd.DataFrame(symbol_data)
            
            # Identify trend
            trend_direction, trend_strength = self._identify_trend(df)
            self.trend_states[symbol] = {
                "direction": trend_direction,
                "strength": trend_strength
            }
            
            # Calculate signals
            signals = self._calculate_signals(df)
            self.last_analysis[symbol] = signals
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Make trading decisions
            if self._should_enter_position(symbol, trend_direction, trend_strength, signals, current_price):
                decisions[symbol] = {
                    "action": "buy" if trend_direction > 0 else "sell",
                    "price": current_price,
                    "size": self.position_size * (1 + trend_strength),
                    "signals": signals,
                    "trend": {
                        "direction": trend_direction,
                        "strength": trend_strength
                    }
                }
            elif self._should_exit_position(symbol, trend_direction, trend_strength, signals, current_price):
                decisions[symbol] = {
                    "action": "sell" if self.current_positions[symbol]["direction"] > 0 else "buy",
                    "price": current_price,
                    "size": self.current_positions[symbol]["size"],
                    "signals": signals,
                    "trend": {
                        "direction": trend_direction,
                        "strength": trend_strength
                    }
                }
        
        return decisions
    
    def update_position(self, symbol: str, position: Dict[str, Any]) -> None:
        """
        Update the current position for a symbol.
        
        Args:
            symbol: Trading symbol
            position: Position information
        """
        if position["size"] == 0:
            self.current_positions.pop(symbol, None)
        else:
            self.current_positions[symbol] = position
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get the current state of the strategy.
        
        Returns:
            Dictionary containing strategy state
        """
        return {
            "current_positions": self.current_positions,
            "trend_states": self.trend_states,
            "last_analysis": self.last_analysis,
            "parameters": {
                "trend_threshold": self.trend_threshold,
                "trend_confirmation_periods": self.trend_confirmation_periods,
                "entry_signal_threshold": self.entry_signal_threshold,
                "exit_signal_threshold": self.exit_signal_threshold,
                "position_size": self.position_size,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "max_positions": self.max_positions
            }
        } 