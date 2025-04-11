"""
Breakout Strategy Agent - Implements breakout trading strategy.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from services.market_data import MarketDataService
from utils.technical_indicators import calculate_bollinger_bands, calculate_atr, calculate_volume_profile

class BreakoutStrategyAgent(BaseAgent):
    """
    Breakout Strategy Agent that implements breakout trading logic.
    
    This agent:
    - Identifies price breakouts from consolidation patterns
    - Uses volume profile and ATR for confirmation
    - Implements proper entry and exit logic
    - Manages position sizing based on breakout strength
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the breakout strategy agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="strategy")
        
        # Strategy parameters
        self.consolidation_period = config.get("consolidation_period", 20)  # days
        self.breakout_threshold = config.get("breakout_threshold", 0.02)  # 2% breakout threshold
        self.volume_threshold = config.get("volume_threshold", 1.5)  # 150% of average volume
        self.entry_signal_threshold = config.get("entry_signal_threshold", 0.7)
        self.exit_signal_threshold = config.get("exit_signal_threshold", 0.3)
        
        # Technical indicator parameters
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2)
        
        self.atr_period = config.get("atr_period", 14)
        self.atr_multiplier = config.get("atr_multiplier", 2)
        
        self.vp_bins = config.get("vp_bins", 20)
        self.vp_threshold = config.get("vp_threshold", 0.7)
        
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
        self.consolidation_ranges = {}
        self.last_analysis = {}
        
        self.logger.info(f"Breakout Strategy Agent initialized with threshold {self.breakout_threshold}")
    
    def _identify_consolidation(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Identify consolidation range and breakout levels.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (support, resistance, breakout_strength)
        """
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
        
        # Calculate volume profile
        price_levels, volume_profile = calculate_volume_profile(
            data['close'], 
            data['volume'], 
            self.vp_bins
        )
        
        # Identify support and resistance
        support = data['low'].rolling(window=self.consolidation_period).min().iloc[-1]
        resistance = data['high'].rolling(window=self.consolidation_period).max().iloc[-1]
        
        # Calculate breakout strength
        range_size = resistance - support
        current_price = data['close'].iloc[-1]
        breakout_strength = 0
        
        if current_price > resistance:
            breakout_strength = (current_price - resistance) / range_size
        elif current_price < support:
            breakout_strength = (support - current_price) / range_size
        
        return support, resistance, breakout_strength
    
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
        
        # Calculate ATR
        atr = calculate_atr(
            data['high'], 
            data['low'], 
            data['close'], 
            self.atr_period
        )
        
        # Calculate volume profile
        price_levels, volume_profile = calculate_volume_profile(
            data['close'], 
            data['volume'], 
            self.vp_bins
        )
        
        # Calculate signal strengths
        bb_signal = 0
        if data['close'].iloc[-1] > upper.iloc[-1]:
            bb_signal = 1.0
        elif data['close'].iloc[-1] < lower.iloc[-1]:
            bb_signal = -1.0
        
        # Calculate volume signal
        avg_volume = data['volume'].rolling(window=self.consolidation_period).mean()
        volume_signal = 1.0 if data['volume'].iloc[-1] > avg_volume.iloc[-1] * self.volume_threshold else 0.0
        
        # Calculate volatility signal
        volatility = atr.iloc[-1] / data['close'].iloc[-1]
        volatility_signal = 1.0 if volatility > self.breakout_threshold else 0.0
        
        # Combine signals
        total_signal = (bb_signal + volume_signal + volatility_signal) / 3.0
        
        return {
            "bollinger": bb_signal,
            "volume": volume_signal,
            "volatility": volatility_signal,
            "total": total_signal
        }
    
    def _should_enter_position(self, 
                             symbol: str, 
                             support: float,
                             resistance: float,
                             breakout_strength: float,
                             signals: Dict[str, float], 
                             price: float) -> bool:
        """
        Determine if we should enter a position.
        
        Args:
            symbol: Trading symbol
            support: Support level
            resistance: Resistance level
            breakout_strength: Breakout strength
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
        
        # Check breakout strength
        if breakout_strength < self.breakout_threshold:
            return False
        
        # Check signal strength
        if signals["total"] < self.entry_signal_threshold:
            return False
        
        # Check if price is outside consolidation range
        if support <= price <= resistance:
            return False
        
        return True
    
    def _should_exit_position(self, 
                            symbol: str, 
                            support: float,
                            resistance: float,
                            breakout_strength: float,
                            signals: Dict[str, float], 
                            price: float) -> bool:
        """
        Determine if we should exit a position.
        
        Args:
            symbol: Trading symbol
            support: Support level
            resistance: Resistance level
            breakout_strength: Breakout strength
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
        
        # Check if price returns to consolidation range
        if support <= price <= resistance:
            return True
        
        # Check exit signals
        if signals["total"] < -self.exit_signal_threshold:
            return True
        
        # Check breakout strength
        if breakout_strength < self.breakout_threshold:
            return True
        
        return False
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the breakout strategy.
        
        Args:
            data: Dictionary of market data
            
        Returns:
            Dictionary of trading decisions
        """
        decisions = {}
        
        for symbol, symbol_data in data.items():
            # Convert to DataFrame
            df = pd.DataFrame(symbol_data)
            
            # Identify consolidation range
            support, resistance, breakout_strength = self._identify_consolidation(df)
            self.consolidation_ranges[symbol] = {
                "support": support,
                "resistance": resistance,
                "breakout_strength": breakout_strength
            }
            
            # Calculate signals
            signals = self._calculate_signals(df)
            self.last_analysis[symbol] = signals
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Make trading decisions
            if self._should_enter_position(symbol, support, resistance, breakout_strength, signals, current_price):
                decisions[symbol] = {
                    "action": "buy" if current_price > resistance else "sell",
                    "price": current_price,
                    "size": self.position_size * (1 + breakout_strength),
                    "signals": signals,
                    "breakout": {
                        "support": support,
                        "resistance": resistance,
                        "strength": breakout_strength
                    }
                }
            elif self._should_exit_position(symbol, support, resistance, breakout_strength, signals, current_price):
                decisions[symbol] = {
                    "action": "sell" if self.current_positions[symbol]["direction"] > 0 else "buy",
                    "price": current_price,
                    "size": self.current_positions[symbol]["size"],
                    "signals": signals,
                    "breakout": {
                        "support": support,
                        "resistance": resistance,
                        "strength": breakout_strength
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
            "consolidation_ranges": self.consolidation_ranges,
            "last_analysis": self.last_analysis,
            "parameters": {
                "consolidation_period": self.consolidation_period,
                "breakout_threshold": self.breakout_threshold,
                "volume_threshold": self.volume_threshold,
                "entry_signal_threshold": self.entry_signal_threshold,
                "exit_signal_threshold": self.exit_signal_threshold,
                "position_size": self.position_size,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "max_positions": self.max_positions
            }
        } 