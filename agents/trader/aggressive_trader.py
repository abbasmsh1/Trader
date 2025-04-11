"""
Aggressive DQN Trader Agent - High-Risk, High-Reward trading strategy.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class AggressiveTraderAgent(DQNTraderAgent):
    """
    Aggressive DQN Trader Agent - High-Risk, High-Reward trading strategy.
    
    This agent takes larger positions with higher turnover, focusing on momentum
    and volatility. It's willing to accept higher drawdowns for potentially larger gains.
    
    Characteristics:
    - Takes larger positions
    - More frequent trading
    - Momentum-focused
    - Tolerates higher drawdowns
    - Seeks higher-volatility assets
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Aggressive DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with aggressive parameters
        self.position_size = config.get("position_size", 0.25)  # Larger position size (25% of available funds)
        self.max_drawdown = config.get("max_drawdown", 0.20)   # Allow higher drawdowns (20%)
        self.stop_loss = config.get("stop_loss", 0.10)         # Wider stop loss (10%)
        self.take_profit = config.get("take_profit", 0.25)     # Higher take profit target (25%)
        
        # Set aggressive personality traits
        self.personality = {
            "risk_appetite": 0.9,       # Very high risk appetite
            "patience": 0.2,            # Low patience (frequent trading)
            "conviction": 0.8,          # High conviction in trades
            "adaptability": 0.7,        # High adaptability to changing conditions
            "contrarian": 0.3           # Primarily momentum-focused, not contrarian
        }
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.9)            # High initial exploration
        self.epsilon_decay = config.get("epsilon_decay", 0.97)  # Slower decay for more exploration
        
        self.logger.info(f"Aggressive DQN Trader {name} initialized with risk appetite {self.personality['risk_appetite']}")
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on momentum and volatility.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
            
        Returns:
            Normalized state array
        """
        # Get base state from parent class
        state = super()._get_state(market_data, symbol)
        
        # If no market data, return the basic state
        if symbol not in market_data:
            return state
        
        # Extract additional features relevant to aggressive trading
        price = market_data[symbol].get("price", 0)
        
        # Enhanced momentum features
        price_change_1h = market_data[symbol].get("price_change_1h", 0)
        price_change_24h = market_data[symbol].get("price_change_24h", 0)
        volume_change_24h = market_data[symbol].get("volume_change_24h", 0)
        
        # Volatility features
        volatility = market_data[symbol].get("volatility", 0.01)
        atr = market_data[symbol].get("atr", volatility * price)  # Average True Range
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to aggressive trading
        if len(state) >= 10:
            # Replace elements that are less important for aggressive trading
            state[2] = np.tanh(price_change_1h * 20)     # Short-term price change
            state[5] = np.tanh(price_change_24h * 10)    # Medium-term price change
            state[6] = np.tanh(volatility * 200)         # Enhanced volatility sensitivity
            state[8] = np.tanh(volume_change_24h * 5)    # Volume momentum
            state[9] = self.personality["risk_appetite"] # Agent's risk appetite
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on higher returns and momentum.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # Amplify rewards for aggressive trading
        value_change = new_value - old_value
        
        # Normalize by portfolio size
        if old_value > 0:
            normalized_change = value_change / old_value
        else:
            normalized_change = 0
        
        # Boost rewards for positive outcomes
        if normalized_change > 0:
            reward *= 1.5  # Amplify positive rewards
        
        # Less penalty for negative outcomes (encourages risk-taking)
        if normalized_change < 0:
            reward *= 0.8  # Reduce negative penalties
        
        # Penalize inaction (holding) if the agent has high conviction
        if action == "hold":
            conviction_factor = self.personality["conviction"]
            reward *= (1.0 - conviction_factor * 0.5)  # Reduce reward for holding
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with aggressive bias toward buying.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to buy
        """
        # Get DQN-based signal
        signal = super()._generate_buy_signal(symbol, market_data)
        
        # If DQN says buy, definitely buy
        if signal:
            return True
        
        # Even if DQN doesn't say buy, sometimes buy anyway based on:
        # 1. Momentum indicators
        # 2. Risk appetite
        if symbol in market_data:
            price_change_1h = market_data[symbol].get("price_change_1h", 0)
            volume_change = market_data[symbol].get("volume_change_24h", 0)
            rsi = market_data[symbol].get("rsi", 50)
            
            # Strong positive momentum conditions
            strong_momentum = (
                price_change_1h > 0.02 and  # >2% hourly gain
                volume_change > 0.5 and     # >50% volume increase
                rsi < 70                     # Not yet overbought
            )
            
            # Buy on strong momentum even if DQN doesn't signal
            if strong_momentum and np.random.random() < self.personality["risk_appetite"]:
                return True
        
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with quick profit-taking behavior.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to sell
        """
        # Get DQN-based signal
        signal = super()._generate_sell_signal(symbol, market_data)
        
        # If DQN says sell, definitely sell
        if signal:
            return True
        
        # Check if we have a position
        position = 0
        entry_price = 0
        if symbol in self.portfolio:
            position = self.portfolio[symbol].get("position", 0)
            entry_price = self.portfolio[symbol].get("entry_price", 0)
        
        # If no position, no need to sell
        if position <= 0 or entry_price <= 0:
            return False
        
        # Get current price
        if symbol in market_data:
            current_price = market_data[symbol].get("price", 0)
            
            # If we have a position, check profit/loss
            if current_price > 0:
                profit_pct = (current_price - entry_price) / entry_price
                
                # Take profit at higher threshold
                if profit_pct >= self.take_profit:
                    return True
                
                # Cut losses at stop loss
                if profit_pct <= -self.stop_loss:
                    return True
                
                # Quick reversal detection - more sensitive to reversals
                price_change_1h = market_data[symbol].get("price_change_1h", 0)
                if profit_pct > 0.05 and price_change_1h < -0.01:
                    # If we're in profit but price is starting to reverse, consider selling
                    sell_probability = self.personality["adaptability"] * 0.7
                    return np.random.random() < sell_probability
        
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate an aggressive-styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        if decision == "buy":
            return f"BUYING {symbol} aggressively to capitalize on market momentum and potential short-term gains."
        elif decision == "sell":
            return f"SELLING {symbol} quickly to lock in profits and free up capital for new opportunities."
        else:
            return f"Holding {symbol} for now, but actively watching for a strong entry or exit opportunity." 