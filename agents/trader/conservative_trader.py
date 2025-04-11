"""
Conservative DQN Trader Agent - Low-Risk, Steady-Gains trading strategy.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class ConservativeTraderAgent(DQNTraderAgent):
    """
    Conservative DQN Trader Agent - Low-Risk, Steady-Gains trading strategy.
    
    This agent takes smaller positions with lower turnover, focusing on value
    and stability. It avoids significant drawdowns, prioritizing capital preservation.
    
    Characteristics:
    - Takes smaller positions
    - Less frequent trading
    - Value-focused
    - Low tolerance for drawdowns
    - Seeks lower-volatility assets
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Conservative DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with conservative parameters
        self.position_size = config.get("position_size", 0.05)  # Smaller position size (5% of available funds)
        self.max_drawdown = config.get("max_drawdown", 0.05)   # Low drawdown tolerance (5%)
        self.stop_loss = config.get("stop_loss", 0.03)         # Tight stop loss (3%)
        self.take_profit = config.get("take_profit", 0.10)     # Reasonable take profit target (10%)
        
        # Set conservative personality traits
        self.personality = {
            "risk_appetite": 0.2,       # Very low risk appetite
            "patience": 0.9,            # High patience (infrequent trading)
            "conviction": 0.6,          # Moderate conviction in trades
            "adaptability": 0.4,        # Lower adaptability to changing conditions
            "contrarian": 0.6           # Slightly contrarian - value-focused
        }
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.5)            # Lower initial exploration
        self.epsilon_decay = config.get("epsilon_decay", 0.99)  # Faster decay for more exploitation
        
        self.logger.info(f"Conservative DQN Trader {name} initialized with risk appetite {self.personality['risk_appetite']}")
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on stability and value.
        
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
        
        # Extract additional features relevant to conservative trading
        price = market_data[symbol].get("price", 0)
        
        # Value and stability features
        ma_50 = market_data[symbol].get("ma_50", price)
        ma_200 = market_data[symbol].get("ma_200", price)
        volatility = market_data[symbol].get("volatility", 0.01)
        rsi = market_data[symbol].get("rsi", 50)
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to conservative trading
        if len(state) >= 10:
            # Replace elements with features more important for conservative trading
            state[2] = np.tanh((price / ma_50 - 1) * 5)     # Price vs 50-day MA (value)
            state[5] = np.tanh((price / ma_200 - 1) * 3)    # Price vs 200-day MA (value)
            state[6] = np.tanh(-volatility * 100)           # Lower volatility preference
            state[8] = np.tanh((rsi - 50) / 25)             # RSI normalized (-1 to 1)
            state[9] = self.personality["risk_appetite"]    # Agent's risk appetite
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on capital preservation and consistency.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # Adjust rewards for conservative trading
        value_change = new_value - old_value
        
        # Normalize by portfolio size
        if old_value > 0:
            normalized_change = value_change / old_value
        else:
            normalized_change = 0
        
        # Less amplification for positive outcomes
        if normalized_change > 0:
            reward *= 1.2  # Slightly increase positive rewards
        
        # Higher penalty for negative outcomes (discourages risk-taking)
        if normalized_change < 0:
            reward *= 1.5  # Increase negative penalties
        
        # Reward holding during uncertain conditions
        if action == "hold":
            patience_factor = self.personality["patience"]
            reward += patience_factor * 0.1  # Small additional reward for patience
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with conservative bias toward value and stability.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to buy
        """
        # Get DQN-based signal
        signal = super()._generate_buy_signal(symbol, market_data)
        
        # Conservative approach: additional value checks before buying
        if signal and symbol in market_data:
            price = market_data[symbol].get("price", 0)
            ma_50 = market_data[symbol].get("ma_50", price)
            ma_200 = market_data[symbol].get("ma_200", price)
            rsi = market_data[symbol].get("rsi", 50)
            volatility = market_data[symbol].get("volatility", 0.01)
            
            # Avoid buying if:
            # 1. Price is significantly above moving averages (>5%)
            # 2. RSI indicates overbought (>70)
            # 3. Volatility is unusually high (>3x average)
            price_to_ma50_ratio = price / ma_50 if ma_50 > 0 else 1
            price_to_ma200_ratio = price / ma_200 if ma_200 > 0 else 1
            avg_volatility = market_data[symbol].get("avg_volatility", volatility)
            volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
            
            high_price = price_to_ma50_ratio > 1.05 or price_to_ma200_ratio > 1.05
            overbought = rsi > 70
            high_volatility = volatility_ratio > 3.0
            
            if high_price or overbought or high_volatility:
                # Risk assessment - still buy sometimes but with lower probability
                if np.random.random() > self.personality["risk_appetite"]:
                    return False  # Avoid buying in risky conditions
        
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with capital preservation focus.
        
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
                
                # Conservative take profit - sell sooner to secure gains
                if profit_pct >= self.take_profit * 0.7:  # Take profit earlier (70% of target)
                    return True
                
                # Conservative stop loss - exit quicker to prevent losses
                if profit_pct <= -self.stop_loss * 0.8:  # Exit at 80% of stop loss
                    return True
                
                # Risk indicators - sell on negative signals
                rsi = market_data[symbol].get("rsi", 50)
                volatility = market_data[symbol].get("volatility", 0.01)
                avg_volatility = market_data[symbol].get("avg_volatility", volatility)
                
                # Sell on warning signs: increasing volatility or overbought conditions
                if (rsi > 75 or volatility > avg_volatility * 2) and profit_pct > 0:
                    # If in profit and seeing warning signs, exit with higher probability
                    sell_probability = self.personality["contrarian"] * 0.8
                    return np.random.random() < sell_probability
        
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a conservative-styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        if decision == "buy":
            return f"Cautiously buying {symbol} at what appears to be fair value with acceptable risk levels."
        elif decision == "sell":
            return f"Selling {symbol} to protect capital and secure profits, avoiding potential downside risk."
        else:
            return f"Holding {symbol} while carefully monitoring market conditions for significant changes." 