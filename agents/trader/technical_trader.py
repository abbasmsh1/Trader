"""
Technical DQN Trader Agent - Chart Patterns and Technical Indicators strategy.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class TechnicalTraderAgent(DQNTraderAgent):
    """
    Technical DQN Trader Agent - Chart Patterns and Technical Indicators strategy.
    
    This agent exclusively uses technical analysis for decision making, focusing
    on indicators, oscillators, and chart patterns. It's more active than average,
    trading on shorter timeframes, and adapting to changing market conditions.
    
    Characteristics:
    - Trades based on technical indicators
    - Medium to high trading frequency
    - Adapts to different market regimes
    - Favors momentum in trending markets
    - Uses oscillators in ranging markets
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Technical DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with technical trading parameters
        self.position_size = config.get("position_size", 0.15)  # Medium position size
        self.max_drawdown = config.get("max_drawdown", 0.15)    # Medium drawdown tolerance
        self.stop_loss = config.get("stop_loss", 0.05)          # Medium stop loss
        self.take_profit = config.get("take_profit", 0.12)      # Medium take profit
        
        # Technical trading specific parameters
        self.rsi_oversold = config.get("rsi_oversold", 30)      # RSI oversold threshold
        self.rsi_overbought = config.get("rsi_overbought", 70)  # RSI overbought threshold
        self.ma_short = config.get("ma_short", 10)              # Short moving average period
        self.ma_medium = config.get("ma_medium", 50)            # Medium moving average period
        self.ma_long = config.get("ma_long", 200)               # Long moving average period
        self.macd_signal = config.get("macd_signal", 9)         # MACD signal period
        
        # Set technical trader personality traits
        self.personality = {
            "risk_appetite": 0.6,       # Medium risk appetite
            "patience": 0.4,            # Lower patience (more active trading)
            "conviction": 0.7,          # High conviction in technical signals
            "adaptability": 0.8,        # High adaptability to market regimes
            "momentum_bias": 0.7        # Preference for momentum indicators
        }
        
        # Technical analysis market regime detection
        self.market_regime = "unknown"  # "trending", "ranging", or "unknown"
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.7)            # More exploration
        self.epsilon_decay = config.get("epsilon_decay", 0.97)  # Medium decay
        
        self.logger.info(f"Technical DQN Trader {name} initialized with conviction {self.personality['conviction']}")
    
    def _detect_market_regime(self, market_data: Dict[str, Any], symbol: str) -> str:
        """
        Detect if market is trending or ranging based on technical indicators.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
            
        Returns:
            Market regime string: "trending", "ranging", or "unknown"
        """
        if symbol not in market_data:
            return "unknown"
            
        # Get data
        atr = market_data[symbol].get("atr", 0)
        adx = market_data[symbol].get("adx", 0)
        
        # Detect regime based on ADX (Avg Directional Index)
        # ADX > 25 typically indicates trending market
        if adx > 25:
            return "trending"
        elif adx < 20:
            return "ranging"
        
        # If ADX is inconclusive, use ATR volatility
        avg_atr = market_data[symbol].get("avg_atr", atr)
        if atr > avg_atr * 1.2:
            return "trending"  # Higher volatility suggests trending
        elif atr < avg_atr * 0.8:
            return "ranging"   # Lower volatility suggests ranging
            
        return "unknown"
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on technical indicators.
        
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
        
        # Determine current market regime
        self.market_regime = self._detect_market_regime(market_data, symbol)
        
        # Extract additional features relevant to technical trading
        price = market_data[symbol].get("price", 0)
        
        # Technical indicators
        ma_short = market_data[symbol].get(f"ma_{self.ma_short}", price)
        ma_medium = market_data[symbol].get(f"ma_{self.ma_medium}", price)
        ma_long = market_data[symbol].get(f"ma_{self.ma_long}", price)
        rsi = market_data[symbol].get("rsi", 50)
        macd = market_data[symbol].get("macd", 0)
        macd_signal = market_data[symbol].get("macd_signal", 0)
        bollinger_upper = market_data[symbol].get("bollinger_upper", price * 1.02)
        bollinger_lower = market_data[symbol].get("bollinger_lower", price * 0.98)
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to technical trading
        if len(state) >= 10:
            # Replace elements with technical indicators
            state[2] = np.tanh((price / ma_short - 1) * 10)   # Price vs short MA
            state[3] = np.tanh((price / ma_medium - 1) * 8)   # Price vs medium MA
            state[4] = np.tanh((price / ma_long - 1) * 5)     # Price vs long MA
            state[5] = np.tanh((rsi - 50) / 25)               # RSI normalized (-1 to 1)
            state[6] = np.tanh(macd * 5)                      # MACD 
            state[7] = np.tanh(macd - macd_signal)            # MACD histogram
            state[8] = np.tanh((price - bollinger_lower) / (bollinger_upper - bollinger_lower) * 2 - 1)  # Bollinger position
            
            # Add market regime context (0 for ranging, 0.5 for unknown, 1 for trending)
            regime_value = 0.5  # unknown
            if self.market_regime == "trending":
                regime_value = 1.0
            elif self.market_regime == "ranging":
                regime_value = 0.0
                
            state[9] = regime_value
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on technical trading metrics.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # Adjust rewards based on market regime
        if self.market_regime == "trending":
            # In trending markets, reward trades aligned with the trend more
            if action != "hold" and new_value > old_value:
                reward *= 1.3  # Boost rewards for successful trades in trending markets
        
        elif self.market_regime == "ranging":
            # In ranging markets, penalize overtrading
            if action != "hold":
                patience_factor = 1 - self.personality["patience"]
                reward -= patience_factor * 0.1  # Small penalty for trading in ranging markets
            else:
                # Reward holding in ranging markets
                reward += 0.05
                
        # Conviction adjustment
        conviction = self.personality["conviction"]
        if new_value > old_value:
            # Higher conviction results in higher rewards for successful trades
            reward *= (1 + 0.2 * conviction)
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with technical analysis focus.
        
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
            
        # Technical analysis approach
        price = market_data[symbol].get("price", 0)
        ma_short = market_data[symbol].get(f"ma_{self.ma_short}", price)
        ma_medium = market_data[symbol].get(f"ma_{self.ma_medium}", price)
        ma_long = market_data[symbol].get(f"ma_{self.ma_long}", price)
        rsi = market_data[symbol].get("rsi", 50)
        macd = market_data[symbol].get("macd", 0)
        macd_signal = market_data[symbol].get("macd_signal", 0)
        bollinger_lower = market_data[symbol].get("bollinger_lower", price * 0.98)
        
        # Different indicators based on market regime
        if self.market_regime == "trending":
            # In trending markets, focus on momentum indicators
            
            # Golden cross (short MA crosses above medium MA)
            short_above_medium = ma_short > ma_medium
            # Both above long MA (established uptrend)
            both_above_long = ma_short > ma_long and ma_medium > ma_long
            # MACD positive and above signal line (bullish)
            macd_bullish = macd > 0 and macd > macd_signal
            
            trend_buy = short_above_medium and both_above_long
            
            # Higher conviction in trending markets
            if trend_buy or macd_bullish:
                conviction_check = np.random.random() < self.personality["conviction"]
                if conviction_check:
                    return True
        
        else:  # Ranging or unknown market
            # In ranging markets, focus on oscillators and mean reversion
            
            # RSI oversold (potential bounce)
            rsi_oversold = rsi < self.rsi_oversold
            # Price near lower Bollinger Band (support)
            price_near_support = price < bollinger_lower * 1.01
            
            # Mean reversion signals
            reversion_buy = rsi_oversold or price_near_support
            
            if reversion_buy:
                # Lower conviction in ranging markets
                conviction_check = np.random.random() < self.personality["conviction"] * 0.8
                if conviction_check:
                    return True
        
        # If technical analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with technical analysis focus.
        
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
            
        # Technical analysis approach
        price = market_data[symbol].get("price", 0)
        ma_short = market_data[symbol].get(f"ma_{self.ma_short}", price)
        ma_medium = market_data[symbol].get(f"ma_{self.ma_medium}", price)
        rsi = market_data[symbol].get("rsi", 50)
        macd = market_data[symbol].get("macd", 0)
        macd_signal = market_data[symbol].get("macd_signal", 0)
        bollinger_upper = market_data[symbol].get("bollinger_upper", price * 1.02)
        
        # Different indicators based on market regime
        if self.market_regime == "trending":
            # In trending markets, sell on trend reversals
            
            # Death cross (short MA crosses below medium MA)
            short_below_medium = ma_short < ma_medium
            # MACD bearish crossover or below zero
            macd_bearish = (macd < 0) or (macd < macd_signal and macd_signal < 0)
            
            trend_sell = short_below_medium and macd_bearish
            
            if trend_sell:
                conviction_check = np.random.random() < self.personality["conviction"]
                if conviction_check:
                    return True
        
        else:  # Ranging or unknown market
            # In ranging markets, sell on overbought conditions
            
            # RSI overbought
            rsi_overbought = rsi > self.rsi_overbought
            # Price near upper Bollinger Band (resistance)
            price_near_resistance = price > bollinger_upper * 0.99
            
            reversion_sell = rsi_overbought or price_near_resistance
            
            if reversion_sell:
                conviction_check = np.random.random() < self.personality["conviction"] * 0.8
                if conviction_check:
                    return True
        
        # If technical analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a technical analysis-styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        market_desc = f"in a {self.market_regime} market"
        
        if decision == "buy":
            if self.market_regime == "trending":
                return f"Technical buy signal for {symbol}: MA alignment and momentum indicators show bullish continuation {market_desc}."
            else:
                return f"Technical buy signal for {symbol}: Oversold conditions and support levels indicate potential reversal {market_desc}."
        elif decision == "sell":
            if self.market_regime == "trending":
                return f"Technical sell signal for {symbol}: MA crossover and momentum shift indicate trend weakening {market_desc}."
            else:
                return f"Technical sell signal for {symbol}: Overbought conditions and resistance levels suggest reversion {market_desc}."
        else:
            return f"Holding {symbol}: Current technical indicators don't show clear trading opportunities {market_desc}." 