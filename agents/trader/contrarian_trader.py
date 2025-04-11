"""
Contrarian DQN Trader Agent - Going against prevailing market sentiment.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class ContrarianTraderAgent(DQNTraderAgent):
    """
    Contrarian DQN Trader Agent - Trading against market sentiment.
    
    This agent deliberately trades against prevailing market sentiment and crowd behavior.
    It buys during periods of extreme fear and sells during periods of excessive greed.
    Designed to capitalize on market overreactions and mean reversion.
    
    Characteristics:
    - Buys during market fear/panic
    - Sells during market euphoria/greed
    - Lower trading frequency
    - Higher conviction on extreme sentiment readings
    - Utilizes sentiment indicators and market breadth
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Contrarian DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with contrarian trading parameters
        self.position_size = config.get("position_size", 0.20)   # Larger position size
        self.max_drawdown = config.get("max_drawdown", 0.25)     # Higher drawdown tolerance
        self.stop_loss = config.get("stop_loss", 0.12)           # Wider stop loss
        self.take_profit = config.get("take_profit", 0.25)       # Higher profit target
        
        # Contrarian specific parameters
        self.fear_threshold = config.get("fear_threshold", 30)   # Extreme fear threshold
        self.greed_threshold = config.get("greed_threshold", 70) # Extreme greed threshold
        self.sentiment_weight = config.get("sentiment_weight", 0.7) # Weight of sentiment in decisions
        self.vix_fear_level = config.get("vix_fear_level", 30)   # VIX fear level
        self.max_crowd_size = config.get("max_crowd_size", 0.7)  # Max crowd following before opposing
        
        # Set contrarian personality traits
        self.personality = {
            "risk_appetite": 0.7,       # Higher risk appetite
            "patience": 0.8,            # High patience (less frequent trading)
            "conviction": 0.8,          # High conviction in contrarian signals
            "adaptability": 0.5,        # Moderate adaptability (sticks to contrarian approach)
            "contrarian_bias": 0.9      # Strong contrarian bias
        }
        
        # Contrarian analysis and tracking
        self.market_sentiment = 50      # 0-100, with 0 being extreme fear, 100 being extreme greed
        self.crowd_consensus = "neutral" # "bullish", "bearish", or "neutral"
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.6)            # Moderate exploration
        self.epsilon_decay = config.get("epsilon_decay", 0.95)  # Moderate decay
        
        self.logger.info(f"Contrarian DQN Trader {name} initialized with contrarian bias {self.personality['contrarian_bias']}")
    
    def _analyze_sentiment(self, market_data: Dict[str, Any], symbol: str) -> None:
        """
        Analyze market sentiment and crowd consensus.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
        """
        if symbol not in market_data:
            self.market_sentiment = 50
            self.crowd_consensus = "neutral"
            return
            
        # Get data
        sentiment_index = market_data[symbol].get("sentiment_index", 50)
        rsi = market_data[symbol].get("rsi", 50)
        vix = market_data[symbol].get("vix", 20)
        put_call_ratio = market_data[symbol].get("put_call_ratio", 1.0)
        
        # Combine indicators to form sentiment index
        # If no sentiment_index is provided, create a synthetic one
        if sentiment_index == 50:
            # Convert RSI to sentiment scale (higher RSI = higher greed)
            rsi_sentiment = rsi
            
            # Convert VIX to sentiment scale (higher VIX = higher fear)
            vix_sentiment = max(0, 100 - vix * 2.5)
            
            # Convert put/call ratio to sentiment (higher ratio = higher fear)
            pc_sentiment = max(0, 100 - put_call_ratio * 50)
            
            # Weighted sentiment calculation
            self.market_sentiment = int(rsi_sentiment * 0.4 + vix_sentiment * 0.3 + pc_sentiment * 0.3)
        else:
            self.market_sentiment = int(sentiment_index)
            
        # Determine crowd consensus
        if self.market_sentiment > self.greed_threshold:
            self.crowd_consensus = "bullish"
        elif self.market_sentiment < self.fear_threshold:
            self.crowd_consensus = "bearish"
        else:
            self.crowd_consensus = "neutral"
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on contrarian indicators.
        
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
        
        # Update sentiment analysis
        self._analyze_sentiment(market_data, symbol)
        
        # Extract additional features relevant to contrarian trading
        price = market_data[symbol].get("price", 0)
        
        # Sentiment and contrarian indicators
        rsi = market_data[symbol].get("rsi", 50)
        vix = market_data[symbol].get("vix", 20)
        put_call_ratio = market_data[symbol].get("put_call_ratio", 1.0)
        market_breadth = market_data[symbol].get("market_breadth", 0) # Positive = more advancers, negative = more decliners
        volatility = market_data[symbol].get("volatility", 0.02)
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to contrarian trading
        if len(state) >= 10:
            # Replace elements with contrarian indicators
            state[2] = np.tanh((50 - self.market_sentiment) / 25)  # Sentiment (negative when greedy)
            state[3] = np.tanh((rsi - 50) / -25)                   # Inverse RSI (negative when overbought)
            state[4] = np.tanh((vix - 20) / 15)                    # VIX deviation from normal
            state[5] = np.tanh((put_call_ratio - 1) * 2)           # Put/call ratio (positive when fearful)
            state[6] = np.tanh(market_breadth * -2)                # Inverse market breadth
            state[7] = np.tanh(volatility * 20)                    # Volatility (higher is better for contrarian)
            
            # Add crowd consensus context (-1 for bearish, 0 for neutral, 1 for bullish)
            consensus_value = 0  # neutral
            if self.crowd_consensus == "bullish":
                consensus_value = -1.0  # Contrarian is negative when crowd is bullish
            elif self.crowd_consensus == "bearish":
                consensus_value = 1.0   # Contrarian is positive when crowd is bearish
                
            state[8] = consensus_value
            state[9] = np.tanh((self.market_sentiment - 50) / -25)  # Inverse sentiment bias
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on contrarian metrics.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # Adjust rewards based on sentiment and action alignment
        if action == "buy":
            # If buying during fear, amplify rewards
            if self.market_sentiment < self.fear_threshold:
                fear_factor = 1 + (1 - self.market_sentiment/100)
                reward *= fear_factor
            
            # If buying during greed, penalize slightly
            elif self.market_sentiment > self.greed_threshold:
                greed_penalty = 0.8
                reward *= greed_penalty
        
        elif action == "sell":
            # If selling during greed, amplify rewards
            if self.market_sentiment > self.greed_threshold:
                greed_factor = 1 + (self.market_sentiment/100)
                reward *= greed_factor
            
            # If selling during fear, penalize slightly
            elif self.market_sentiment < self.fear_threshold:
                fear_penalty = 0.8
                reward *= fear_penalty
        
        # Conviction adjustment for extreme sentiment readings
        if self.market_sentiment < 20 or self.market_sentiment > 80:
            # Higher conviction during extreme sentiment
            conviction = self.personality["conviction"] * 1.2
            if new_value > old_value:
                # Higher rewards for successful contrarian trades
                reward *= (1 + 0.3 * conviction)
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with contrarian focus.
        
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
            
        # Update sentiment analysis
        self._analyze_sentiment(market_data, symbol)
            
        # Contrarian approach
        # Extreme fear or bearish consensus are buy signals
        extreme_fear = self.market_sentiment < self.fear_threshold
        bearish_crowd = self.crowd_consensus == "bearish"
        
        # Get more data
        rsi = market_data[symbol].get("rsi", 50)
        vix = market_data[symbol].get("vix", 20)
        put_call_ratio = market_data[symbol].get("put_call_ratio", 1.0)
        
        # Additional contrarian buy signals
        rsi_oversold = rsi < 30
        high_vix = vix > self.vix_fear_level
        high_put_call = put_call_ratio > 1.2
        
        # Combine signals - more weight to sentiment and crowd consensus
        sentiment_signal = extreme_fear or bearish_crowd
        indicator_signal = rsi_oversold or high_vix or high_put_call
        
        # Contrarian buy logic
        contrarian_buy = (self.sentiment_weight * int(sentiment_signal) + 
                         (1 - self.sentiment_weight) * int(indicator_signal) > 0.5)
        
        # Higher conviction during extreme fear
        if contrarian_buy and self.market_sentiment < 20:
            conviction_check = np.random.random() < self.personality["conviction"] * 1.2
            if conviction_check:
                return True
        
        # Standard conviction during moderate fear
        elif contrarian_buy:
            conviction_check = np.random.random() < self.personality["conviction"]
            if conviction_check:
                return True
        
        # If contrarian analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with contrarian focus.
        
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
            
        # Update sentiment analysis
        self._analyze_sentiment(market_data, symbol)
            
        # Contrarian approach
        # Extreme greed or bullish consensus are sell signals
        extreme_greed = self.market_sentiment > self.greed_threshold
        bullish_crowd = self.crowd_consensus == "bullish"
        
        # Get more data
        rsi = market_data[symbol].get("rsi", 50)
        vix = market_data[symbol].get("vix", 20)
        put_call_ratio = market_data[symbol].get("put_call_ratio", 1.0)
        
        # Additional contrarian sell signals
        rsi_overbought = rsi > 70
        low_vix = vix < 15  # Complacency
        low_put_call = put_call_ratio < 0.8  # More calls than puts (bullish)
        
        # Combine signals - more weight to sentiment and crowd consensus
        sentiment_signal = extreme_greed or bullish_crowd
        indicator_signal = rsi_overbought or low_vix or low_put_call
        
        # Contrarian sell logic
        contrarian_sell = (self.sentiment_weight * int(sentiment_signal) + 
                          (1 - self.sentiment_weight) * int(indicator_signal) > 0.5)
        
        # Higher conviction during extreme greed
        if contrarian_sell and self.market_sentiment > 80:
            conviction_check = np.random.random() < self.personality["conviction"] * 1.2
            if conviction_check:
                return True
        
        # Standard conviction during moderate greed
        elif contrarian_sell:
            conviction_check = np.random.random() < self.personality["conviction"]
            if conviction_check:
                return True
        
        # If contrarian analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a contrarian-styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        sentiment_desc = "neutral"
        if self.market_sentiment < 30:
            sentiment_desc = "fearful"
        elif self.market_sentiment > 70:
            sentiment_desc = "greedy"
        
        if decision == "buy":
            if self.market_sentiment < self.fear_threshold:
                return f"Contrarian buy signal for {symbol}: Market sentiment is {sentiment_desc} ({self.market_sentiment}/100) - buying while others are fearful."
            else:
                return f"Buy signal for {symbol}: Despite moderate sentiment ({self.market_sentiment}/100), technical indicators suggest an oversold condition."
        elif decision == "sell":
            if self.market_sentiment > self.greed_threshold:
                return f"Contrarian sell signal for {symbol}: Market sentiment is {sentiment_desc} ({self.market_sentiment}/100) - selling while others are greedy."
            else:
                return f"Sell signal for {symbol}: Despite moderate sentiment ({self.market_sentiment}/100), technical indicators suggest an overbought condition."
        else:
            return f"Holding {symbol}: Market sentiment ({self.market_sentiment}/100) is not extreme enough to trigger a contrarian trade." 