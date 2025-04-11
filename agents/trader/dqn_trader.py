"""
DQN Trader Agent - Deep Q-Network based trader for reinforcement learning trading.
"""

import numpy as np
import random
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from agents.trader.base_trader import BaseTraderAgent

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

class DQNTraderAgent(BaseTraderAgent):
    """
    Deep Q-Network (DQN) based trader agent.
    
    This agent uses reinforcement learning with a neural network to make trading decisions.
    It learns from experience by recording state, action, reward, next_state tuples and
    periodically training its neural network on these experiences.
    
    The DQN approach enables the agent to capture complex patterns in market data and
    adapt its strategy over time based on actual trading results.
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # DQN specific configuration
        self.state_size = config.get("state_size", 10)
        self.action_size = 3  # 0=hold, 1=buy, 2=sell
        self.batch_size = config.get("batch_size", 32)
        self.gamma = config.get("gamma", 0.95)  # Discount factor
        self.epsilon = config.get("epsilon", 1.0)  # Exploration rate
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.memory_size = config.get("memory_size", 2000)
        self.update_target_frequency = config.get("update_target_frequency", 100)
        
        # Risk management specific to DQN
        self.risk_factor = config.get("risk_factor", 0.5)  # 0.0 to 1.0
        self.max_drawdown = config.get("max_drawdown", 0.10)  # 10% max drawdown
        self.position_sizing = config.get("position_sizing", "fixed")  # fixed, kelly, risk_parity
        
        # Personality traits (to be overridden by subclasses)
        self.personality = {
            "risk_appetite": 0.5,       # 0.0 to 1.0 (low to high)
            "patience": 0.5,            # 0.0 to 1.0 (low to high)
            "conviction": 0.5,          # 0.0 to 1.0 (low to high)
            "adaptability": 0.5,        # 0.0 to 1.0 (low to high)
            "contrarian": 0.5           # 0.0 to 1.0 (follower to contrarian)
        }
        
        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Step counter for target network updates
        self.step_counter = 0
        
        # Initialize neural networks
        self.model = None
        self.target_model = None
        
        if HAS_TENSORFLOW:
            self._build_model()
            self._build_target_model()
        else:
            self.logger.warning("TensorFlow not available. DQN agent will use fallback strategy.")
    
    def _build_model(self):
        """Build the neural network model for DQN."""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.model = model
        
    def _build_target_model(self):
        """Build the target neural network model for DQN."""
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
    
    def _update_target_model(self):
        """Update the target model weights with the main model weights."""
        self.target_model.set_weights(self.model.get_weights())
        self.logger.info("Target model updated")
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features from market data.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
            
        Returns:
            Normalized state array
        """
        # Extract relevant features for the state
        if symbol not in market_data:
            # Return zero state if data not available
            return np.zeros(self.state_size)
        
        # Basic features (to be enhanced by subclasses)
        price = market_data[symbol].get("price", 0)
        volume = market_data[symbol].get("volume", 0)
        
        # Price indicators
        price_sma_10 = market_data[symbol].get("sma_10", price)
        price_sma_30 = market_data[symbol].get("sma_30", price)
        price_ema_10 = market_data[symbol].get("ema_10", price)
        
        # Momentum indicators
        rsi = market_data[symbol].get("rsi", 50) / 100.0  # Normalize to 0-1
        macd = market_data[symbol].get("macd", 0)
        macd_signal = market_data[symbol].get("macd_signal", 0)
        
        # Volatility
        volatility = market_data[symbol].get("volatility", 0.01)
        
        # Current position (normalized)
        position = 0
        if symbol in self.portfolio:
            max_position = self.personal_wallet.get_balance(self.personal_wallet.base_currency) / price
            position = self.portfolio[symbol].get("position", 0) / max_position if max_position > 0 else 0
        
        # Combine into state vector
        state = np.array([
            price / (price_sma_30 + 1e-10) - 1,  # Price relative to 30-day SMA
            price / (price_sma_10 + 1e-10) - 1,  # Price relative to 10-day SMA
            price / (price_ema_10 + 1e-10) - 1,  # Price relative to 10-day EMA
            rsi,                                 # RSI (already normalized)
            macd / (price + 1e-10),              # MACD relative to price
            macd_signal / (price + 1e-10),       # MACD signal relative to price
            np.tanh(volatility * 100),           # Normalized volatility
            position,                            # Current position
            np.log1p(volume) / 15,               # Normalized volume
            self.personality["risk_appetite"]    # Agent's risk appetite
        ])
        
        return state
    
    def memorize(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state vector
            
        Returns:
            Selected action (0=hold, 1=buy, 2=sell)
        """
        # Fallback to random action if TensorFlow not available
        if not HAS_TENSORFLOW:
            return random.randrange(self.action_size)
        
        # Exploration: choose random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: choose best action based on model
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train the model on a batch of experiences from memory."""
        if len(self.memory) < self.batch_size or not HAS_TENSORFLOW:
            return
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        targets = np.zeros((self.batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            if done:
                target = reward
            else:
                # Use target network for more stable Q-value estimates
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            
            # Get current Q-values and update the target for the chosen action
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            targets[i] = target_f[0]
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon for less exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _process_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an update cycle for the DQN agent.
        
        Args:
            data: Input data for processing, including market data
            
        Returns:
            Results of processing
        """
        market_data = data.get("market_data", {})
        
        if not market_data:
            self.logger.warning("No market data provided for update")
            return {"result": "skipped", "reason": "no_data"}
        
        # Update portfolio with latest prices
        self._update_portfolio(market_data)
        
        # Process for each symbol
        results = {}
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            # Get current state
            current_state = self._get_state(market_data, symbol)
            
            # Choose action
            action = self.act(current_state)
            
            # Map action to trading decision
            decision = "hold"
            if action == 1:  # Buy
                decision = "buy"
            elif action == 2:  # Sell
                decision = "sell"
            
            # Store decision
            results[symbol] = {"action": decision}
            
            # Execute decision if active
            if self.active:
                # Get current state values for reward calculation
                old_portfolio_value = self._calculate_portfolio_value()
                
                # Execute the decision
                execution_result = self._execute_decision(symbol, decision, market_data)
                results[symbol]["executed"] = execution_result.get("executed", False)
                
                # Calculate reward
                new_portfolio_value = self._calculate_portfolio_value()
                reward = self._calculate_reward(old_portfolio_value, new_portfolio_value, decision)
                
                # Get next state after execution
                next_state = self._get_state(market_data, symbol)
                
                # Store experience
                done = False  # Not used in this continuous trading scenario
                self.memorize(current_state, action, reward, next_state, done)
                
                # Increment step counter
                self.step_counter += 1
                
                # Periodically train the model
                if len(self.memory) >= self.batch_size:
                    self.replay()
                
                # Update target model periodically
                if self.step_counter % self.update_target_frequency == 0:
                    self._update_target_model()
        
        # Update metrics
        self._update_metrics()
        
        return {
            "decisions": results,
            "metrics": self.performance_metrics,
            "portfolio": self.portfolio
        }
    
    def _execute_decision(self, symbol: str, decision: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading decision for a specific symbol.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            market_data: Market data dictionary
            
        Returns:
            Dictionary with execution results
        """
        result = {"executed": False, "action": decision}
        
        if decision == "hold":
            return result
        
        # Skip if no personal wallet or no market data
        if not self.personal_wallet or symbol not in market_data:
            return result
        
        # Get price and position info
        price = market_data[symbol].get("price", 0)
        position = self.portfolio[symbol].get("position", 0)
        
        try:
            if decision == "buy":
                # Calculate position size based on risk and personality
                base_currency = self.personal_wallet.base_currency
                available_balance = self.personal_wallet.get_balance(base_currency)
                
                # Adjust trade amount based on risk appetite
                risk_adjusted_position_size = self.position_size * (0.5 + self.personality["risk_appetite"] * 0.5)
                
                # Limit position size based on max drawdown
                max_position_pct = min(risk_adjusted_position_size, self.max_drawdown / self.risk_factor)
                
                # Calculate trade amount
                trade_amount = available_balance * max_position_pct
                
                # Check if trade amount meets minimum requirement
                min_trade_value = self.config.get("min_trade_value", 5.0)
                if trade_amount < min_trade_value:
                    self.logger.info(f"Buy amount too small for {symbol}: {trade_amount:.2f} {base_currency} < {min_trade_value:.2f} {base_currency}")
                    result["reason"] = "amount_too_small"
                    return result
                
                # Calculate quantity based on price
                quantity = trade_amount / price
                
                # Execute the trade
                trade_result = self.personal_wallet.add_trade(
                    trade_type="buy",
                    from_currency=base_currency,
                    to_currency=symbol.split('/')[0],  # Extract base currency from symbol
                    from_amount=trade_amount,
                    to_amount=quantity,
                    price=price,
                    fee=trade_amount * 0.001  # 0.1% fee
                )
                
                # Update portfolio tracking
                self._update_portfolio_after_trade(symbol, "buy", quantity, price)
                
                # Mark as executed
                result["executed"] = True
                result["details"] = {
                    "price": price,
                    "quantity": quantity,
                    "value": trade_amount
                }
                
                self.logger.info(f"Executed BUY for {symbol}: {quantity:.6f} @ {price:.2f}")
                
            elif decision == "sell" and position > 0:
                # Calculate quantity to sell (all of current position)
                quantity = position
                trade_currency = symbol.split('/')[0]  # Extract base currency from symbol
                
                # Calculate trade amount
                trade_amount = quantity * price
                
                # Check if trade amount meets minimum requirement
                min_trade_value = self.config.get("min_trade_value", 5.0)
                if trade_amount < min_trade_value:
                    self.logger.info(f"Sell amount too small for {symbol}: {trade_amount:.2f} < {min_trade_value:.2f}")
                    result["reason"] = "amount_too_small"
                    return result
                
                # Execute the trade
                trade_result = self.personal_wallet.add_trade(
                    trade_type="sell",
                    from_currency=trade_currency,
                    to_currency=self.personal_wallet.base_currency,
                    from_amount=quantity,
                    to_amount=trade_amount,
                    price=price,
                    fee=trade_amount * 0.001  # 0.1% fee
                )
                
                # Update portfolio tracking
                self._update_portfolio_after_trade(symbol, "sell", quantity, price)
                
                # Mark as executed
                result["executed"] = True
                result["details"] = {
                    "price": price,
                    "quantity": quantity,
                    "value": trade_amount
                }
                
                self.logger.info(f"Executed SELL for {symbol}: {quantity:.6f} @ {price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing {decision} for {symbol}: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate the total portfolio value.
        
        Returns:
            Total portfolio value
        """
        if not self.personal_wallet:
            return 0.0
        
        return self.personal_wallet.get_total_value()
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward for the reinforcement learning agent.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Base reward on portfolio value change
        value_change = new_value - old_value
        
        # Normalize by portfolio size
        if old_value > 0:
            normalized_change = value_change / old_value
        else:
            normalized_change = 0
        
        # Base reward
        reward = normalized_change * 100  # Scale up for better learning signal
        
        # Personality-based reward adjustments
        if action == "buy":
            # Reward buying on dips if contrarian
            if normalized_change < 0:
                reward *= (1.0 - self.personality["contrarian"])
            # Penalize buying on up moves if contrarian
            else:
                reward *= (1.0 + self.personality["contrarian"] * 0.5)
        
        elif action == "sell":
            # Reward selling on up moves if momentum-based
            if normalized_change > 0:
                reward *= (1.0 - self.personality["contrarian"])
            # Penalize selling on dips if momentum-based
            else:
                reward *= (1.0 + self.personality["contrarian"] * 0.5)
        
        # Patience factor - reduce reward for frequent trading
        if action != "hold" and self.last_trade_time:
            time_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds() / 3600  # hours
            patience_factor = min(1.0, time_since_last_trade / (24 * self.personality["patience"]))
            reward *= patience_factor
        
        return reward
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent for persistence.
        
        Returns:
            Dictionary representing agent state
        """
        # Get base state from parent class
        state = super().get_state()
        
        # Add DQN-specific state
        state["dqn"] = {
            "epsilon": self.epsilon,
            "step_counter": self.step_counter,
            "personality": self.personality
        }
        
        # Note: we don't save the neural network or memory
        # In a production system, you would save/load the model weights
        
        return state
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load agent state from the provided dictionary.
        
        Args:
            state: Dictionary with agent state
            
        Returns:
            Success flag
        """
        # Load base state from parent class
        super().load_state(state)
        
        # Load DQN-specific state
        if "dqn" in state:
            dqn_state = state["dqn"]
            self.epsilon = dqn_state.get("epsilon", self.epsilon)
            self.step_counter = dqn_state.get("step_counter", 0)
            self.personality = dqn_state.get("personality", self.personality)
        
        return True
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal for the given symbol based on DQN decision.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to buy
        """
        # Get current state
        state = self._get_state(market_data, symbol)
        
        # Choose action
        action = self.act(state)
        
        # Return True if action is buy (1)
        return action == 1
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal for the given symbol based on DQN decision.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to sell
        """
        # Get current state
        state = self._get_state(market_data, symbol)
        
        # Choose action
        action = self.act(state)
        
        # Return True if action is sell (2)
        return action == 2

    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a natural language explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        # Override in each personality subclass to provide more specific explanations
        if decision == "buy":
            return f"The DQN model predicted buying {symbol} would optimize returns based on current market conditions."
        elif decision == "sell":
            return f"The DQN model predicted selling {symbol} would optimize returns based on current market conditions."
        else:
            return f"The DQN model determined holding {symbol} is the optimal action in current market conditions." 