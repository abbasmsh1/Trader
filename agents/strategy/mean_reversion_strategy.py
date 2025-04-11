"""
Mean Reversion Strategy Agent - Implements mean reversion trading strategy.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from services.market_data import MarketDataService
from utils.technical_indicators import calculate_bollinger_bands, calculate_rsi, calculate_stochastic

class MeanReversionStrategyAgent(BaseAgent):
    """
    Mean Reversion Strategy Agent that implements mean reversion trading logic.
    
    This agent:
    - Identifies overbought and oversold conditions
    - Uses Bollinger Bands, RSI, and Stochastic for confirmation
    - Implements proper entry and exit logic
    - Manages position sizing based on deviation from mean
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the mean reversion strategy agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="strategy")
        
        # Strategy parameters
        self.mean_period = config.get("mean_period", 20)  # days
        self.deviation_threshold = config.get("deviation_threshold", 0.02)  # 2% deviation
        self.entry_signal_threshold = config.get("entry_signal_threshold", 0.7)
        self.exit_signal_threshold = config.get("exit_signal_threshold", 0.3)
        
        # Technical indicator parameters
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2)
        
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        
        self.stoch_k_period = config.get("stoch_k_period", 14)
        self.stoch_d_period = config.get("stoch_d_period", 3)
        self.stoch_overbought = config.get("stoch_overbought", 80)
        self.stoch_oversold = config.get("stoch_oversold", 20)
        
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
        self.mean_values = {}
        self.last_analysis = {}
        
        self.logger.info(f"Mean Reversion Strategy Agent initialized with threshold {self.deviation_threshold}")
    
    def _calculate_mean_deviation(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate mean value and current deviation.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (mean_value, deviation)
        """
        # Calculate mean value
        mean_value = data['close'].rolling(window=self.mean_period).mean().iloc[-1]
        
        # Calculate current deviation
        current_price = data['close'].iloc[-1]
        deviation = (current_price - mean_value) / mean_value
        
        return mean_value, deviation
    
    def _calculate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trading signals using technical indicators.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary of signal strengths
        """
        # Calculate Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(
            data['close'], 
            self.bb_period, 
            self.bb_std
        )
        
        # Calculate RSI
        rsi = calculate_rsi(data['close'], self.rsi_period)
        
        # Calculate Stochastic
        k_line, d_line = calculate_stochastic(
            data['high'], 
            data['low'], 
            data['close'],
            self.stoch_k_period,
            self.stoch_d_period
        )
        
        # Calculate signal strengths
        bb_signal = 0
        if data['close'].iloc[-1] > upper.iloc[-1]:
            bb_signal = -1.0
        elif data['close'].iloc[-1] < lower.iloc[-1]:
            bb_signal = 1.0
        
        # Calculate RSI signal
        rsi_signal = 0
        if rsi.iloc[-1] < self.rsi_oversold:
            rsi_signal = 1.0
        elif rsi.iloc[-1] > self.rsi_overbought:
            rsi_signal = -1.0
        
        # Calculate Stochastic signal
        stoch_signal = 0
        if k_line.iloc[-1] < self.stoch_oversold and d_line.iloc[-1] < self.stoch_oversold:
            stoch_signal = 1.0
        elif k_line.iloc[-1] > self.stoch_overbought and d_line.iloc[-1] > self.stoch_overbought:
            stoch_signal = -1.0
        
        # Combine signals
        total_signal = (bb_signal + rsi_signal + stoch_signal) / 3.0
        
        return {
            "bollinger": bb_signal,
            "rsi": rsi_signal,
            "stochastic": stoch_signal,
            "total": total_signal
        }
    
    def _should_enter_position(self, 
                             symbol: str, 
                             mean_value: float,
                             deviation: float,
                             signals: Dict[str, float], 
                             price: float) -> bool:
        """
        Determine if we should enter a position.
        
        Args:
            symbol: Trading symbol
            mean_value: Current mean value
            deviation: Current deviation from mean
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
        
        # Check deviation threshold
        if abs(deviation) < self.deviation_threshold:
            return False
        
        # Check signal strength
        if abs(signals["total"]) < self.entry_signal_threshold:
            return False
        
        # Check if signals align with deviation
        if deviation * signals["total"] < 0:
            return False
        
        return True
    
    def _should_exit_position(self, 
                            symbol: str, 
                            mean_value: float,
                            deviation: float,
                            signals: Dict[str, float], 
                            price: float) -> bool:
        """
        Determine if we should exit a position.
        
        Args:
            symbol: Trading symbol
            mean_value: Current mean value
            deviation: Current deviation from mean
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
        
        # Check if price has reverted to mean
        if abs(deviation) < self.deviation_threshold:
            return True
        
        # Check exit signals
        if abs(signals["total"]) < self.exit_signal_threshold:
            return True
        
        # Check if signals indicate reversal
        if deviation * signals["total"] < 0:
            return True
        
        return False
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the mean reversion strategy.
        
        Args:
            data: Dictionary of market data
            
        Returns:
            Dictionary of trading decisions
        """
        decisions = {}
        
        for symbol, symbol_data in data.items():
            # Convert to DataFrame
            df = pd.DataFrame(symbol_data)
            
            # Calculate mean and deviation
            mean_value, deviation = self._calculate_mean_deviation(df)
            self.mean_values[symbol] = {
                "mean": mean_value,
                "deviation": deviation
            }
            
            # Calculate signals
            signals = self._calculate_signals(df)
            self.last_analysis[symbol] = signals
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Make trading decisions
            if self._should_enter_position(symbol, mean_value, deviation, signals, current_price):
                decisions[symbol] = {
                    "action": "buy" if deviation < 0 else "sell",
                    "price": current_price,
                    "size": self.position_size * (1 + abs(deviation)),
                    "signals": signals,
                    "mean_reversion": {
                        "mean": mean_value,
                        "deviation": deviation
                    }
                }
            elif self._should_exit_position(symbol, mean_value, deviation, signals, current_price):
                decisions[symbol] = {
                    "action": "sell" if self.current_positions[symbol]["direction"] > 0 else "buy",
                    "price": current_price,
                    "size": self.current_positions[symbol]["size"],
                    "signals": signals,
                    "mean_reversion": {
                        "mean": mean_value,
                        "deviation": deviation
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
            "mean_values": self.mean_values,
            "last_analysis": self.last_analysis,
            "parameters": {
                "mean_period": self.mean_period,
                "deviation_threshold": self.deviation_threshold,
                "entry_signal_threshold": self.entry_signal_threshold,
                "exit_signal_threshold": self.exit_signal_threshold,
                "position_size": self.position_size,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "max_positions": self.max_positions
            }
        } 