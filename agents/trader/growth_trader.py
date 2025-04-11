"""
Growth DQN Trader Agent - Focusing on companies with strong growth potential.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class GrowthTraderAgent(DQNTraderAgent):
    """
    Growth DQN Trader Agent - Focusing on high-growth companies.
    
    This agent seeks out companies with strong growth characteristics,
    emphasizing revenue growth, earnings growth, and market expansion potential.
    It prioritizes future growth over current valuation, willing to pay premium
    prices for companies with exceptional growth trajectories.
    
    Characteristics:
    - Medium to high trading frequency
    - Emphasis on growth metrics over valuation
    - Higher volatility tolerance
    - Focus on innovation and market disruption
    - Preference for emerging sectors and technologies
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Growth DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with growth-focused parameters
        self.position_size = config.get("position_size", 0.15)     # Standard position size
        self.max_drawdown = config.get("max_drawdown", 0.25)       # Higher drawdown tolerance
        self.stop_loss = config.get("stop_loss", 0.15)             # Wider stop loss for growth
        self.take_profit = config.get("take_profit", 0.35)         # Higher profit target
        
        # Growth specific parameters
        self.revenue_growth_min = config.get("revenue_growth_min", 20)  # Minimum revenue growth % YoY
        self.earnings_growth_min = config.get("earnings_growth_min", 15)  # Minimum earnings growth % YoY
        self.growth_premium = config.get("growth_premium", 1.5)   # Premium willing to pay for growth
        self.innovation_score_min = config.get("innovation_score_min", 70)  # Minimum innovation score (0-100)
        self.tam_growth_min = config.get("tam_growth_min", 10)  # Minimum total addressable market growth %
        self.growth_score_threshold = config.get("growth_score_threshold", 0.65)  # Minimum growth score
        
        # Set growth-focused personality traits
        self.personality = {
            "risk_appetite": 0.75,      # Higher risk appetite
            "patience": 0.55,           # Moderate patience (still needs time for growth)
            "conviction": 0.70,         # Good conviction once growth story established
            "adaptability": 0.80,       # High adaptability to new trends
            "innovation_bias": 0.85     # High bias towards innovation and disruption
        }
        
        # Growth analysis tracking
        self.growth_scores = {}           # Dictionary to track growth scores by symbol
        self.growth_trajectories = {}     # Dictionary to track growth trajectories (accelerating/decelerating)
        self.tam_coverage = {}            # Dictionary to track total addressable market coverage
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.5)             # Higher exploration for growth opportunities
        self.epsilon_decay = config.get("epsilon_decay", 0.97)  # Moderate decay
        
        self.logger.info(f"Growth DQN Trader {name} initialized with innovation bias {self.personality['innovation_bias']}")
    
    def _calculate_growth_metrics(self, market_data: Dict[str, Any], symbol: str) -> float:
        """
        Calculate growth metrics and determine growth potential.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
            
        Returns:
            Growth score from 0 to 1, where higher values indicate stronger growth potential
        """
        if symbol not in market_data:
            # If we have a previously calculated score, use it
            return self.growth_scores.get(symbol, 0.0)
            
        # Extract growth metrics (if available in market data)
        price = market_data[symbol].get("price", 0)
        if price <= 0:
            return 0.0  # Can't calculate growth metrics with invalid price
            
        # Get growth data
        revenue_growth = market_data[symbol].get("revenue_growth", 0)  # YoY revenue growth %
        earnings_growth = market_data[symbol].get("earnings_growth", 0)  # YoY earnings growth %
        projected_growth = market_data[symbol].get("projected_growth", 0)  # Forward growth projection %
        innovation_score = market_data[symbol].get("innovation_score", 50)  # Innovation/disruption score
        tam_growth = market_data[symbol].get("tam_growth", 0)  # Total addressable market growth
        market_share_trend = market_data[symbol].get("market_share_trend", 0)  # Market share momentum
        
        # Additional metrics
        peg_ratio = market_data[symbol].get("peg_ratio", 2.0)  # Price/Earnings to Growth ratio
        sales_growth_q = market_data[symbol].get("sales_growth_q", 0)  # QoQ sales growth %
        r_and_d_to_revenue = market_data[symbol].get("r_and_d_to_revenue", 0)  # R&D as % of revenue
        
        # Calculate growth acceleration/deceleration
        prev_revenue_growth = market_data[symbol].get("prev_revenue_growth", revenue_growth)
        growth_acceleration = revenue_growth - prev_revenue_growth
        
        # Calculate growth scores for each metric (0 to 1 scale, higher is better growth)
        revenue_score = min(1, revenue_growth / self.revenue_growth_min) if self.revenue_growth_min > 0 else 0
        earnings_score = min(1, earnings_growth / self.earnings_growth_min) if self.earnings_growth_min > 0 else 0
        projection_score = min(1, projected_growth / self.revenue_growth_min) if self.revenue_growth_min > 0 else 0
        
        # Innovation and market potential scores
        innovation_score_norm = innovation_score / 100  # Normalize to 0-1
        tam_score = min(1, tam_growth / self.tam_growth_min) if self.tam_growth_min > 0 else 0
        
        # PEG ratio score - lower is better for growth at reasonable price
        peg_score = max(0, min(1, 2 - peg_ratio)) if peg_ratio > 0 else 0
        
        # R&D investment score - higher R&D investment is better for future growth
        rd_score = min(1, r_and_d_to_revenue / 0.15)  # Scale to 15% of revenue
        
        # Market share momentum score
        market_share_score = max(0, min(1, (market_share_trend + 5) / 10))  # Scale from -5% to +5%
        
        # Calculate overall growth score with emphasis on revenue growth and innovation
        financial_growth_weight = 0.45
        innovation_weight = 0.30
        market_weight = 0.15
        valuation_weight = 0.10
        
        growth_score = (
            financial_growth_weight * (0.5 * revenue_score + 0.3 * earnings_score + 0.2 * projection_score) +
            innovation_weight * (0.7 * innovation_score_norm + 0.3 * rd_score) +
            market_weight * (0.6 * tam_score + 0.4 * market_share_score) +
            valuation_weight * peg_score
        )
        
        # Determine growth trajectory (accelerating or decelerating)
        if growth_acceleration > 2:  # More than 2% acceleration
            self.growth_trajectories[symbol] = "accelerating"
        elif growth_acceleration < -2:  # More than 2% deceleration
            self.growth_trajectories[symbol] = "decelerating"
        else:
            self.growth_trajectories[symbol] = "stable"
            
        # Estimate TAM coverage
        market_share = market_data[symbol].get("market_share", 0)
        self.tam_coverage[symbol] = market_share
        
        # Save calculated metrics for later use
        self.growth_scores[symbol] = growth_score
        
        return growth_score
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on growth indicators.
        
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
        
        # Calculate growth metrics
        growth_score = self._calculate_growth_metrics(market_data, symbol)
        
        # Extract additional features relevant to growth trading
        price = market_data[symbol].get("price", 0)
        
        # Growth indicators
        revenue_growth = market_data[symbol].get("revenue_growth", 0)
        earnings_growth = market_data[symbol].get("earnings_growth", 0)
        projected_growth = market_data[symbol].get("projected_growth", 0)
        innovation_score = market_data[symbol].get("innovation_score", 50) / 100  # Normalize to 0-1
        
        # Market indicators
        market_share = market_data[symbol].get("market_share", 0)
        tam_growth = market_data[symbol].get("tam_growth", 0)
        
        # Valuation and momentum
        peg_ratio = market_data[symbol].get("peg_ratio", 2.0)
        price_momentum = market_data[symbol].get("price_momentum", 0)
        relative_strength = market_data[symbol].get("relative_strength", 0)
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to growth trading
        if len(state) >= 10:
            # Replace elements with growth indicators
            state[2] = np.tanh(revenue_growth / 25)    # Revenue growth normalized
            state[3] = np.tanh(earnings_growth / 20)   # Earnings growth normalized
            state[4] = np.tanh(projected_growth / 25)  # Future growth projection
            
            # Innovation and market features
            state[5] = np.tanh((innovation_score - 0.5) * 4)  # Innovation score centered and amplified
            state[6] = np.tanh(tam_growth / 15)           # TAM growth potential
            
            # Valuation in context of growth
            state[7] = np.tanh((1.5 - peg_ratio) * 2)     # PEG ratio (negative when overvalued for growth)
            
            # Momentum features
            state[8] = np.tanh(price_momentum * 5)        # Price momentum amplified
            
            # Overall growth score and trajectory
            trajectory_factor = 0.0
            if symbol in self.growth_trajectories:
                if self.growth_trajectories[symbol] == "accelerating":
                    trajectory_factor = 0.3
                elif self.growth_trajectories[symbol] == "decelerating":
                    trajectory_factor = -0.3
                    
            state[9] = np.tanh((growth_score - 0.5) * 4 + trajectory_factor)  # Growth score centered and amplified
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on growth metrics and potential.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # For growth investors, focus on potential for future gains
        # Adjust rewards based on action and growth metrics
        if self.last_symbol in self.growth_scores:
            growth_score = self.growth_scores[self.last_symbol]
            trajectory = self.growth_trajectories.get(self.last_symbol, "stable")
            
            if action == "buy":
                # If buying high-growth assets, provide additional reward
                if growth_score > self.growth_score_threshold:
                    growth_boost = 1 + (growth_score * 0.4)  # Up to 40% boost for high growth scores
                    reward *= growth_boost
                    
                    # Additional boost for accelerating growth
                    if trajectory == "accelerating":
                        reward *= 1.15  # 15% boost for accelerating growth
                
                # If buying low-growth assets, reduce reward
                elif growth_score < self.growth_score_threshold * 0.7:
                    growth_penalty = max(0.6, growth_score / self.growth_score_threshold)
                    reward *= growth_penalty
            
            elif action == "sell":
                # If selling high-growth assets with accelerating growth, reduce reward
                if growth_score > self.growth_score_threshold and trajectory == "accelerating":
                    patience_factor = self.personality["patience"]
                    reward *= (1 - (patience_factor * 0.2))  # Up to 20% reduction based on patience
                
                # If selling decelerating growth assets, increase reward
                elif trajectory == "decelerating":
                    reward *= 1.15  # 15% boost for avoiding decelerating growth
        
        # Apply risk appetite adjustment - growth investors accept more volatility
        risk_appetite = self.personality["risk_appetite"]
        if new_value < old_value:
            # Less penalty for short-term losses (higher risk tolerance)
            penalty_reduction = risk_appetite * 0.2
            reward *= (1 + penalty_reduction)
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with growth focus.
        
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
            
        # Calculate growth metrics if not already calculated
        if symbol not in self.growth_scores:
            self._calculate_growth_metrics(market_data, symbol)
            
        # Get growth metrics
        growth_score = self.growth_scores.get(symbol, 0.0)
        trajectory = self.growth_trajectories.get(symbol, "stable")
        
        # Growth approach - buy high-growth assets
        strong_growth = growth_score > self.growth_score_threshold
        accelerating = trajectory == "accelerating"
        
        # Get additional indicators
        price_momentum = market_data[symbol].get("price_momentum", 0)
        rsi = market_data[symbol].get("rsi", 50)
        
        # Momentum and technical indicators supportive of growth
        positive_momentum = price_momentum > 0
        not_overbought = rsi < 70  # Not extremely overbought
        
        # Combine signals for growth-based buy decision
        growth_signal = strong_growth and (accelerating or positive_momentum)
        technical_signal = positive_momentum and not_overbought
        
        # Weight the signals based on personality
        innovation_bias = self.personality["innovation_bias"]
        growth_weight = 0.6 + (innovation_bias * 0.2)  # 0.6 to 0.8 based on innovation bias
        technical_weight = 1 - growth_weight
        
        growth_buy = ((growth_weight * int(growth_signal) + 
                      technical_weight * int(technical_signal)) > 0.5)
        
        # Apply conviction based on strength of growth signal
        if growth_buy:
            # Higher conviction for stronger growth signals, especially with acceleration
            conviction_boost = 1.0
            if accelerating:
                conviction_boost += 0.2
                
            conviction_factor = self.personality["conviction"] * conviction_boost
            conviction_check = np.random.random() < conviction_factor
            if conviction_check:
                return True
        
        # If growth analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with growth focus.
        
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
            
        # Calculate growth metrics if not already calculated
        if symbol not in self.growth_scores:
            self._calculate_growth_metrics(market_data, symbol)
            
        # Get growth metrics
        growth_score = self.growth_scores.get(symbol, 0.0)
        trajectory = self.growth_trajectories.get(symbol, "stable")
        
        # Growth approach - sell when growth story deteriorates
        weak_growth = growth_score < (self.growth_score_threshold * 0.7)
        decelerating = trajectory == "decelerating"
        
        # Get price and entry data for this position
        current_price = market_data[symbol].get("price", 0)
        entry_price = self.portfolio[symbol].get("entry_price", current_price)
        
        # Calculate price appreciation
        if entry_price > 0:
            price_appreciation = (current_price - entry_price) / entry_price
        else:
            price_appreciation = 0
            
        # Get additional indicators
        price_momentum = market_data[symbol].get("price_momentum", 0)
        rsi = market_data[symbol].get("rsi", 50)
        
        # Momentum and technical indicators that might signal selling
        negative_momentum = price_momentum < -0.02  # Significant negative momentum
        overbought = rsi > 75  # Extremely overbought
        
        # Growth sell signal - deteriorating fundamentals or excessive valuation
        growth_signal = weak_growth or decelerating
        technical_signal = negative_momentum or overbought
        
        # Special case: Take profit on massive run-ups even if growth still good
        take_profit_signal = price_appreciation > self.take_profit and overbought
        
        # Weight the signals based on personality
        innovation_bias = self.personality["innovation_bias"]
        growth_weight = 0.6 + (innovation_bias * 0.2)  # 0.6 to 0.8 based on innovation bias
        technical_weight = 1 - growth_weight
        
        growth_sell = ((growth_weight * int(growth_signal) + 
                       technical_weight * int(technical_signal)) > 0.5) or take_profit_signal
        
        # Apply patience - less likely to sell too quickly on temporary setbacks
        holding_time = self.portfolio[symbol].get("holding_time", 0)
        min_hold_time = 10  # Minimum holding periods for growth to play out
        
        if holding_time < min_hold_time and not weak_growth:
            # Lower probability of selling growth assets too early
            patience_check = np.random.random() < (1 - self.personality["patience"])
            if not patience_check:
                return False
                
        # Apply conviction if selling due to growth signals
        if growth_sell:
            # Higher conviction for clear deterioration signals
            conviction_factor = self.personality["conviction"]
            if decelerating:
                conviction_factor *= 1.2  # 20% boost for decelerating growth
                
            conviction_check = np.random.random() < conviction_factor
            if conviction_check:
                return True
        
        # If growth analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a growth-investing styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        # Get growth metrics
        growth_score = self.growth_scores.get(symbol, 0.0)
        trajectory = self.growth_trajectories.get(symbol, "stable")
        
        # Format growth description
        growth_desc = "modest growth potential"
        if growth_score > 0.8:
            growth_desc = "exceptional growth potential"
        elif growth_score > 0.65:
            growth_desc = "strong growth potential"
        elif growth_score < 0.4:
            growth_desc = "limited growth potential"
        elif growth_score < 0.2:
            growth_desc = "poor growth outlook"
            
        # Format trajectory description
        trajectory_desc = ""
        if trajectory == "accelerating":
            trajectory_desc = " with accelerating momentum"
        elif trajectory == "decelerating":
            trajectory_desc = " but showing deceleration"
            
        # Generate explanation based on decision
        if decision == "buy":
            if growth_score > self.growth_score_threshold:
                return f"Growth buy signal for {symbol}: Company shows {growth_desc}{trajectory_desc}. Investing in future market leaders."
            else:
                return f"Buy signal for {symbol}: While not a top-tier growth story, recent developments suggest improving growth trajectory and market position."
        elif decision == "sell":
            if growth_score < self.growth_score_threshold:
                return f"Growth sell signal for {symbol}: Company shows {growth_desc}{trajectory_desc}. Capital can be redeployed to higher-growth opportunities."
            else:
                return f"Sell signal for {symbol}: Despite {growth_desc}, technical indicators suggest taking profits or risk factors have emerged."
        else:
            return f"Holding {symbol}: Company maintains {growth_desc}{trajectory_desc}. Continuing to monitor growth metrics and market positioning." 