"""
Momentum DQN Trader Agent - Following market trends and momentum.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class MomentumTraderAgent(DQNTraderAgent):
    """
    Momentum DQN Trader Agent - Riding market trends and momentum.
    
    This agent follows market trends, buying assets that are rising in price
    and selling those that are falling. It focuses on identifying and capitalizing
    on price momentum across different timeframes.
    
    Characteristics:
    - Follows market trends rather than fighting them
    - Medium to high trading frequency
    - Utilizes moving averages and momentum indicators
    - Aims to capture large trends while managing risk
    - Higher sensitivity to breakouts and momentum shifts
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Momentum DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with momentum-focused parameters
        self.position_size = config.get("position_size", 0.15)     # Standard position size
        self.max_drawdown = config.get("max_drawdown", 0.15)       # Lower drawdown tolerance
        self.stop_loss = config.get("stop_loss", 0.08)             # Tighter stop loss for momentum
        self.take_profit = config.get("take_profit", 0.20)         # Higher profit target
        
        # Momentum specific parameters
        self.short_ma_period = config.get("short_ma_period", 9)    # Short moving average period
        self.medium_ma_period = config.get("medium_ma_period", 21) # Medium moving average period
        self.long_ma_period = config.get("long_ma_period", 50)     # Long moving average period
        self.trend_strength_threshold = config.get("trend_strength_threshold", 0.6)  # ADX threshold
        self.momentum_threshold = config.get("momentum_threshold", 0.02)  # Minimum momentum
        self.breakout_volume_factor = config.get("breakout_volume_factor", 1.5)  # Volume increase for breakouts
        
        # Set momentum-focused personality traits
        self.personality = {
            "risk_appetite": 0.65,      # Above average risk appetite
            "patience": 0.4,            # Lower patience (more frequent trading)
            "conviction": 0.7,          # Good conviction once trend established
            "adaptability": 0.8,        # High adaptability to changing trends
            "trend_bias": 0.85          # Strong bias towards following trends
        }
        
        # Trend analysis and tracking
        self.trend_direction = "sideways"  # "up", "down", or "sideways"
        self.trend_strength = 0.0         # 0.0 to 1.0 (weak to strong trend)
        self.in_breakout = False          # Flag for price breakouts
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.4)         # Lower exploration (rely more on trends)
        self.epsilon_decay = config.get("epsilon_decay", 0.98)  # Slower decay
        
        self.logger.info(f"Momentum DQN Trader {name} initialized with trend bias {self.personality['trend_bias']}")
    
    def _analyze_trend(self, market_data: Dict[str, Any], symbol: str) -> None:
        """
        Analyze current market trend direction and strength.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
        """
        if symbol not in market_data:
            self.trend_direction = "sideways"
            self.trend_strength = 0.0
            self.in_breakout = False
            return
            
        # Get moving averages
        short_ma = market_data[symbol].get(f"ma_{self.short_ma_period}", 0)
        medium_ma = market_data[symbol].get(f"ma_{self.medium_ma_period}", 0)
        long_ma = market_data[symbol].get(f"ma_{self.long_ma_period}", 0)
        
        # Get price and technical indicators
        price = market_data[symbol].get("price", 0)
        adx = market_data[symbol].get("adx", 15)  # Average Directional Index
        
        # Get volume data
        volume = market_data[symbol].get("volume", 0)
        avg_volume = market_data[symbol].get("avg_volume", 1)
        
        # Determine trend direction
        if price > short_ma > medium_ma > long_ma:
            self.trend_direction = "up"
            self.trend_strength = min(adx / 100, 1.0)  # Normalize ADX to 0-1
        elif price < short_ma < medium_ma < long_ma:
            self.trend_direction = "down"
            self.trend_strength = min(adx / 100, 1.0)  # Normalize ADX to 0-1
        else:
            # Check if close to changing trend
            if short_ma > medium_ma and medium_ma < long_ma:
                self.trend_direction = "potential_up"
                self.trend_strength = min(adx / 200, 0.5)  # Lower strength for potential trends
            elif short_ma < medium_ma and medium_ma > long_ma:
                self.trend_direction = "potential_down"
                self.trend_strength = min(adx / 200, 0.5)  # Lower strength for potential trends
            else:
                self.trend_direction = "sideways"
                self.trend_strength = max(0, min(adx / 200, 0.3))  # Very low strength for sideways
        
        # Check for breakout conditions
        self.in_breakout = False
        if volume > (avg_volume * self.breakout_volume_factor):
            # Volume spike
            price_change = market_data[symbol].get("daily_change", 0)
            if abs(price_change) > self.momentum_threshold:
                self.in_breakout = True
                # Strengthen trend signal during breakouts
                self.trend_strength = min(self.trend_strength * 1.5, 1.0)
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on momentum indicators.
        
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
        
        # Update trend analysis
        self._analyze_trend(market_data, symbol)
        
        # Extract additional features relevant to momentum trading
        price = market_data[symbol].get("price", 0)
        
        # Momentum indicators
        rsi = market_data[symbol].get("rsi", 50)
        macd = market_data[symbol].get("macd", 0)
        macd_signal = market_data[symbol].get("macd_signal", 0)
        macd_hist = market_data[symbol].get("macd_hist", 0)
        obv = market_data[symbol].get("obv", 0)  # On Balance Volume
        obv_ma = market_data[symbol].get("obv_ma", 0)  # OBV Moving Average
        
        # Get moving averages
        short_ma = market_data[symbol].get(f"ma_{self.short_ma_period}", price)
        medium_ma = market_data[symbol].get(f"ma_{self.medium_ma_period}", price)
        long_ma = market_data[symbol].get(f"ma_{self.long_ma_period}", price)
        
        # Calculate price distance from moving averages (normalized)
        price_vs_short = (price - short_ma) / (short_ma + 1e-8)
        price_vs_medium = (price - medium_ma) / (medium_ma + 1e-8)
        price_vs_long = (price - long_ma) / (long_ma + 1e-8)
        
        # Calculate momentum from moving averages
        short_medium_momentum = (short_ma - medium_ma) / (medium_ma + 1e-8)
        medium_long_momentum = (medium_ma - long_ma) / (long_ma + 1e-8)
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to momentum trading
        if len(state) >= 10:
            # Replace elements with momentum indicators
            state[2] = np.tanh(price_vs_short * 10)       # Price vs short MA 
            state[3] = np.tanh(price_vs_medium * 8)       # Price vs medium MA
            state[4] = np.tanh(price_vs_long * 5)         # Price vs long MA
            
            # MACD related features
            state[5] = np.tanh(macd * 10)                 # MACD value
            state[6] = np.tanh(macd_hist * 15)            # MACD histogram (momentum)
            
            # Volume and RSI features
            state[7] = np.tanh((rsi - 50) / 25)           # RSI momentum
            state[8] = np.tanh((obv - obv_ma) * 0.01)     # OBV momentum
            
            # Trend context
            trend_value = 0.0  # Neutral
            if self.trend_direction == "up" or self.trend_direction == "potential_up":
                trend_value = self.trend_strength
            elif self.trend_direction == "down" or self.trend_direction == "potential_down":
                trend_value = -self.trend_strength
                
            state[9] = np.tanh(trend_value * 2)  # Trend direction and strength
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on momentum metrics and trend following.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # Adjust rewards based on trend alignment
        if action == "buy":
            # If buying during uptrend, amplify rewards
            if self.trend_direction == "up":
                trend_factor = 1 + self.trend_strength
                reward *= trend_factor
            
            # If buying during downtrend, penalize
            elif self.trend_direction == "down":
                trend_penalty = max(0.6, 1 - self.trend_strength)
                reward *= trend_penalty
                
            # Bonus for buying during breakouts
            if self.in_breakout and self.trend_direction in ["up", "potential_up"]:
                reward *= 1.2
        
        elif action == "sell":
            # If selling during downtrend, amplify rewards
            if self.trend_direction == "down":
                trend_factor = 1 + self.trend_strength
                reward *= trend_factor
            
            # If selling during uptrend, penalize
            elif self.trend_direction == "up":
                trend_penalty = max(0.6, 1 - self.trend_strength)
                reward *= trend_penalty
                
            # Bonus for selling during breakouts in downtrends
            if self.in_breakout and self.trend_direction in ["down", "potential_down"]:
                reward *= 1.2
        
        # Conviction adjustment for strong trends
        if self.trend_strength > self.trend_strength_threshold:
            # Higher conviction during strong trends
            conviction = self.personality["conviction"] * (1 + 0.5 * self.trend_strength)
            if new_value > old_value:
                # Higher rewards for successful trend-following trades
                reward *= (1 + 0.2 * conviction)
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with momentum focus.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to buy
        """
        # Get DQN-based signal
        signal = super()._generate_buy_signal(symbol, market_data)
        
        # If we have no market data, rely on DQN signal
        if symbol not in market_data:
            return signal
            
        # Update trend analysis
        self._analyze_trend(market_data, symbol)
            
        # Momentum approach - buy on uptrends
        uptrend = self.trend_direction in ["up", "potential_up"]
        strong_trend = self.trend_strength > self.trend_strength_threshold
        
        # Get more data
        rsi = market_data[symbol].get("rsi", 50)
        macd = market_data[symbol].get("macd", 0)
        macd_hist = market_data[symbol].get("macd_hist", 0)
        
        # Additional momentum buy signals
        positive_momentum = macd > 0 and macd_hist > 0
        rsi_momentum = rsi > 50 and rsi < 70  # Not overbought but showing strength
        
        # Get moving averages for crossover detection
        short_ma = market_data[symbol].get(f"ma_{self.short_ma_period}", 0)
        medium_ma = market_data[symbol].get(f"ma_{self.medium_ma_period}", 0)
        
        # Detect golden cross (short term MA crosses above medium term MA)
        golden_cross = short_ma > medium_ma and market_data[symbol].get("prev_short_ma", 0) <= market_data[symbol].get("prev_medium_ma", 0)
        
        # Combine signals
        trend_signal = uptrend and (strong_trend or self.in_breakout)
        indicator_signal = positive_momentum or rsi_momentum or golden_cross
        
        # Momentum buy logic - more weight to trend direction and breakouts
        trend_weight = self.personality["trend_bias"]
        momentum_buy = (trend_weight * int(trend_signal) + 
                       (1 - trend_weight) * int(indicator_signal) > 0.5)
        
        # Higher conviction during strong uptrends or breakouts
        if momentum_buy and (strong_trend or self.in_breakout):
            conviction_boost = 1.2 if self.in_breakout else 1.1
            conviction_check = np.random.random() < self.personality["conviction"] * conviction_boost
            if conviction_check:
                return True
        
        # Standard conviction during other momentum signals
        elif momentum_buy:
            conviction_check = np.random.random() < self.personality["conviction"]
            if conviction_check:
                return True
        
        # If momentum analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with momentum focus.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to sell
        """
        # Get DQN-based signal
        signal = super()._generate_sell_signal(symbol, market_data)
        
        # Check if we have a position
        position = 0
        if symbol in self.portfolio:
            position = self.portfolio[symbol].get("position", 0)
        
        # If no position, no need to sell
        if position <= 0:
            return False
            
        # If we have no market data, rely on DQN signal
        if symbol not in market_data:
            return signal
            
        # Update trend analysis
        self._analyze_trend(market_data, symbol)
            
        # Momentum approach - sell on downtrends
        downtrend = self.trend_direction in ["down", "potential_down"]
        strong_trend = self.trend_strength > self.trend_strength_threshold
        
        # Get more data
        rsi = market_data[symbol].get("rsi", 50)
        macd = market_data[symbol].get("macd", 0)
        macd_hist = market_data[symbol].get("macd_hist", 0)
        
        # Additional momentum sell signals
        negative_momentum = macd < 0 and macd_hist < 0
        rsi_weakness = rsi < 50 and rsi > 30  # Not oversold but showing weakness
        
        # Get moving averages for crossover detection
        short_ma = market_data[symbol].get(f"ma_{self.short_ma_period}", 0)
        medium_ma = market_data[symbol].get(f"ma_{self.medium_ma_period}", 0)
        
        # Detect death cross (short term MA crosses below medium term MA)
        death_cross = short_ma < medium_ma and market_data[symbol].get("prev_short_ma", 0) >= market_data[symbol].get("prev_medium_ma", 0)
        
        # Take profit logic - sell on extreme RSI in uptrends
        take_profit_signal = False
        if self.trend_direction == "up" and rsi > 75:
            take_profit_signal = True
        
        # Combine signals
        trend_signal = downtrend and (strong_trend or self.in_breakout)
        indicator_signal = negative_momentum or rsi_weakness or death_cross or take_profit_signal
        
        # Momentum sell logic - more weight to trend direction and breakouts
        trend_weight = self.personality["trend_bias"]
        momentum_sell = (trend_weight * int(trend_signal) + 
                        (1 - trend_weight) * int(indicator_signal) > 0.5)
        
        # Higher conviction during strong downtrends or breakouts
        if momentum_sell and (strong_trend or self.in_breakout):
            conviction_boost = 1.2 if self.in_breakout else 1.1
            conviction_check = np.random.random() < self.personality["conviction"] * conviction_boost
            if conviction_check:
                return True
        
        # Standard conviction during other momentum signals
        elif momentum_sell:
            conviction_check = np.random.random() < self.personality["conviction"]
            if conviction_check:
                return True
        
        # If momentum analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a momentum-styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        # Create a description of the current trend
        trend_desc = "sideways market"
        strength_desc = "weak"
        
        if self.trend_strength > 0.7:
            strength_desc = "strong"
        elif self.trend_strength > 0.4:
            strength_desc = "moderate"
            
        if self.trend_direction == "up":
            trend_desc = f"{strength_desc} uptrend"
        elif self.trend_direction == "down":
            trend_desc = f"{strength_desc} downtrend"
        elif self.trend_direction == "potential_up":
            trend_desc = f"potential uptrend forming"
        elif self.trend_direction == "potential_down":
            trend_desc = f"potential downtrend forming"
            
        # Add breakout information if applicable
        breakout_desc = ""
        if self.in_breakout:
            breakout_desc = " with volume breakout"
            
        # Generate explanation based on decision
        if decision == "buy":
            if self.trend_direction in ["up", "potential_up"]:
                return f"Momentum buy signal for {symbol}: Riding {trend_desc}{breakout_desc}. The trend is our friend."
            else:
                return f"Buy signal for {symbol}: Despite {trend_desc}, momentum indicators suggest a potential reversal to the upside."
        elif decision == "sell":
            if self.trend_direction in ["down", "potential_down"]:
                return f"Momentum sell signal for {symbol}: Exiting during {trend_desc}{breakout_desc}. Following the path of least resistance."
            else:
                return f"Sell signal for {symbol}: Taking profits or cutting losses as momentum indicators suggest weakening in the current {trend_desc}."
        else:
            return f"Holding {symbol}: Current {trend_desc} doesn't provide a strong momentum signal. Waiting for clearer direction." 