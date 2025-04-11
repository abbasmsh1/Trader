"""
Value DQN Trader Agent - Finding undervalued assets through fundamental analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from agents.trader.dqn_trader import DQNTraderAgent

class ValueTraderAgent(DQNTraderAgent):
    """
    Value DQN Trader Agent - Finding undervalued assets based on fundamentals.
    
    This agent focuses on identifying assets that are trading below their intrinsic value,
    using metrics such as P/E ratios, book value, cash flow analysis, and other
    fundamental indicators to make value-oriented investment decisions.
    
    Characteristics:
    - Lower trading frequency (longer holding periods)
    - High emphasis on fundamental analysis
    - Conservative risk management
    - Contrarian approach in overvalued markets
    - Ignores short-term price fluctuations
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the Value DQN trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id)
        
        # Override default settings with value-focused parameters
        self.position_size = config.get("position_size", 0.25)     # Larger positions
        self.max_drawdown = config.get("max_drawdown", 0.25)       # Higher drawdown tolerance
        self.stop_loss = config.get("stop_loss", 0.20)             # Wider stop loss for value
        self.take_profit = config.get("take_profit", 0.50)         # Higher profit target
        
        # Value specific parameters
        self.pe_threshold = config.get("pe_threshold", 15.0)       # P/E ratio threshold
        self.pb_threshold = config.get("pb_threshold", 1.5)        # Price-to-book threshold
        self.ev_ebitda_threshold = config.get("ev_ebitda_threshold", 8.0)  # EV/EBITDA threshold
        self.dividend_yield_min = config.get("dividend_yield_min", 0.02)   # Minimum dividend yield
        self.margin_of_safety = config.get("margin_of_safety", 0.30)       # Margin of safety percentage
        self.undervalue_score_threshold = config.get("undervalue_threshold", 0.6)  # Minimum undervalue score
        
        # Set value-focused personality traits
        self.personality = {
            "risk_appetite": 0.35,      # Lower risk appetite
            "patience": 0.85,           # High patience (long-term holding)
            "conviction": 0.80,         # High conviction once value is found
            "adaptability": 0.30,       # Lower adaptability (sticks to strategy)
            "contrarian": 0.70          # High contrarian tendency
        }
        
        # Value analysis tracking
        self.value_scores = {}          # Dictionary to track value scores by symbol
        self.intrinsic_values = {}      # Dictionary to track estimated intrinsic values
        self.underpriced = {}           # Dictionary to track underpriced assets
        
        # Adjust DQN parameters
        self.epsilon = config.get("epsilon", 0.3)         # Lower exploration (rely more on fundamentals)
        self.epsilon_decay = config.get("epsilon_decay", 0.99)  # Slower decay
        
        self.logger.info(f"Value DQN Trader {name} initialized with margin of safety {self.margin_of_safety}")
    
    def _calculate_value_metrics(self, market_data: Dict[str, Any], symbol: str) -> float:
        """
        Calculate value metrics and determine if an asset is undervalued.
        
        Args:
            market_data: Market data dictionary
            symbol: Trading symbol
            
        Returns:
            Value score from 0 to 1, where higher values indicate more undervalued
        """
        if symbol not in market_data:
            # If we have a previously calculated score, use it
            return self.value_scores.get(symbol, 0.0)
            
        # Extract fundamental metrics (if available in market data)
        price = market_data[symbol].get("price", 0)
        if price <= 0:
            return 0.0  # Can't calculate value metrics with invalid price
            
        # Get fundamental data
        pe_ratio = market_data[symbol].get("pe_ratio", 30)  # Price-to-Earnings
        pb_ratio = market_data[symbol].get("pb_ratio", 3)   # Price-to-Book
        ev_ebitda = market_data[symbol].get("ev_ebitda", 12)  # Enterprise Value to EBITDA
        dividend_yield = market_data[symbol].get("dividend_yield", 0)  # Dividend Yield
        fcf_yield = market_data[symbol].get("fcf_yield", 0)  # Free Cash Flow Yield
        roe = market_data[symbol].get("roe", 0)  # Return on Equity
        debt_to_equity = market_data[symbol].get("debt_to_equity", 1)  # Debt-to-Equity
        current_ratio = market_data[symbol].get("current_ratio", 1)  # Current Ratio
        
        # Calculate value scores for each metric (0 to 1 scale, higher is better value)
        pe_score = max(0, 1 - (pe_ratio / (self.pe_threshold * 2))) if pe_ratio > 0 else 0
        pb_score = max(0, 1 - (pb_ratio / (self.pb_threshold * 2))) if pb_ratio > 0 else 0
        ev_ebitda_score = max(0, 1 - (ev_ebitda / (self.ev_ebitda_threshold * 2))) if ev_ebitda > 0 else 0
        
        # Dividend score - higher dividends are better for value
        dividend_score = min(1, dividend_yield / (self.dividend_yield_min * 2))
        
        # FCF yield score - higher FCF yield is better
        fcf_score = min(1, fcf_yield / 0.1)  # Scale up to 10% FCF yield
        
        # Quality metrics - good companies at reasonable prices
        quality_score = 0.0
        if roe > 15:
            quality_score += 0.5
        if debt_to_equity < 1:
            quality_score += 0.25
        if current_ratio > 1.5:
            quality_score += 0.25
        
        # Calculate overall value score with emphasis on PE, PB and EV/EBITDA
        value_metrics_weight = 0.6
        quality_weight = 0.2
        income_weight = 0.2
        
        value_score = (
            value_metrics_weight * (0.4 * pe_score + 0.3 * pb_score + 0.3 * ev_ebitda_score) +
            quality_weight * quality_score +
            income_weight * (0.7 * fcf_score + 0.3 * dividend_score)
        )
        
        # Estimate intrinsic value based on fundamental data
        earnings_per_share = market_data[symbol].get("eps", price / pe_ratio if pe_ratio > 0 else 0)
        book_value_per_share = market_data[symbol].get("book_value", price / pb_ratio if pb_ratio > 0 else 0)
        
        # Simple intrinsic value estimate (averaging multiple methods)
        pe_based_value = earnings_per_share * self.pe_threshold if earnings_per_share > 0 else 0
        pb_based_value = book_value_per_share * self.pb_threshold if book_value_per_share > 0 else 0
        
        # Combine intrinsic value estimates with appropriate weights
        if pe_based_value > 0 and pb_based_value > 0:
            intrinsic_value = (0.7 * pe_based_value + 0.3 * pb_based_value)
        elif pe_based_value > 0:
            intrinsic_value = pe_based_value
        elif pb_based_value > 0:
            intrinsic_value = pb_based_value
        else:
            intrinsic_value = 0
        
        # Apply margin of safety
        safe_value = intrinsic_value * (1 - self.margin_of_safety)
        
        # Determine if underpriced
        self.underpriced[symbol] = price < safe_value if safe_value > 0 else False
        
        # Save calculated metrics for later use
        self.value_scores[symbol] = value_score
        self.intrinsic_values[symbol] = intrinsic_value
        
        return value_score
    
    def _get_state(self, market_data: Dict[str, Any], symbol: str) -> np.ndarray:
        """
        Extract and normalize state features with emphasis on value indicators.
        
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
        
        # Calculate value metrics
        value_score = self._calculate_value_metrics(market_data, symbol)
        
        # Extract additional features relevant to value trading
        price = market_data[symbol].get("price", 0)
        
        # Value indicators
        pe_ratio = market_data[symbol].get("pe_ratio", 30)
        pb_ratio = market_data[symbol].get("pb_ratio", 3)
        ev_ebitda = market_data[symbol].get("ev_ebitda", 12)
        dividend_yield = market_data[symbol].get("dividend_yield", 0)
        
        # Quality indicators
        roe = market_data[symbol].get("roe", 0)
        debt_to_equity = market_data[symbol].get("debt_to_equity", 1)
        
        # Get moving averages for long term trends
        ma_200 = market_data[symbol].get("ma_200", price)
        price_vs_ma_200 = (price - ma_200) / (ma_200 + 1e-8)
        
        # Normalize and replace some elements in the state vector to emphasize
        # features more relevant to value trading
        if len(state) >= 10:
            # Replace elements with value indicators
            state[2] = np.tanh((self.pe_threshold - pe_ratio) / self.pe_threshold) if pe_ratio > 0 else 0
            state[3] = np.tanh((self.pb_threshold - pb_ratio) / self.pb_threshold) if pb_ratio > 0 else 0
            state[4] = np.tanh((self.ev_ebitda_threshold - ev_ebitda) / self.ev_ebitda_threshold) if ev_ebitda > 0 else 0
            
            # Dividend and quality features
            state[5] = np.tanh(dividend_yield * 10)  # Scale up dividend yield
            state[6] = np.tanh((roe - 10) / 10)      # ROE relative to 10% benchmark
            
            # Price relative to intrinsic value
            intrinsic_value = self.intrinsic_values.get(symbol, price)
            if intrinsic_value > 0:
                price_to_intrinsic = price / intrinsic_value
                state[7] = np.tanh((1 - price_to_intrinsic) * 2)  # Negative when overvalued, positive when undervalued
            
            # Long term trend context
            state[8] = np.tanh(price_vs_ma_200 * 2)  # Price relative to 200-day MA
            
            # Overall value score
            state[9] = np.tanh((value_score - 0.5) * 4)  # Center around 0.5 and amplify
        
        return state
    
    def _calculate_reward(self, old_value: float, new_value: float, action: str) -> float:
        """
        Calculate reward with an emphasis on value metrics and long-term performance.
        
        Args:
            old_value: Portfolio value before action
            new_value: Portfolio value after action
            action: Action taken
            
        Returns:
            Reward value
        """
        # Get base reward from parent class
        reward = super()._calculate_reward(old_value, new_value, action)
        
        # For value investors, short-term losses are less concerning if fundamentals are strong
        # Adjust rewards based on action and value metrics
        if self.last_symbol in self.value_scores:
            value_score = self.value_scores[self.last_symbol]
            is_underpriced = self.underpriced.get(self.last_symbol, False)
            
            if action == "buy":
                # If buying undervalued assets, provide additional reward even if short-term loss
                if is_underpriced:
                    value_boost = 1 + (value_score * 0.5)  # Up to 50% boost for high value scores
                    reward *= value_boost
                
                # If buying overvalued assets, reduce reward
                elif value_score < self.undervalue_score_threshold:
                    value_penalty = max(0.5, value_score)  # Penalize more for lower value scores
                    reward *= value_penalty
            
            elif action == "sell":
                # If selling undervalued assets too early, reduce reward
                if is_underpriced and new_value < old_value:
                    patience_factor = self.personality["patience"]
                    reward *= (1 - (patience_factor * 0.3))  # Up to 30% reduction based on patience
                
                # If selling overvalued assets for profit, increase reward
                elif not is_underpriced and new_value > old_value:
                    reward *= 1.2  # 20% boost for selling overvalued assets
        
        # Apply conviction adjustment
        conviction = self.personality["conviction"]
        if new_value > old_value:
            reward *= (1 + 0.1 * conviction)  # Amplify positive outcomes with conviction
        
        return reward
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal with value focus.
        
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
            
        # Calculate value metrics if not already calculated
        if symbol not in self.value_scores:
            self._calculate_value_metrics(market_data, symbol)
            
        # Get value metrics
        value_score = self.value_scores.get(symbol, 0.0)
        is_underpriced = self.underpriced.get(symbol, False)
        
        # Value approach - buy undervalued assets
        strong_value = value_score > self.undervalue_score_threshold
        margin_of_safety_met = is_underpriced
        
        # Combine signals for value-based buy decision
        value_buy = strong_value and margin_of_safety_met
        
        # Add contrarian element - more likely to buy in market downturns
        market_sentiment = market_data.get("market_sentiment", 0)  # -1 to 1 scale
        contrarian_opportunity = market_sentiment < -0.3  # Market fear creates opportunities
        
        if contrarian_opportunity and strong_value:
            contrarian_boost = self.personality["contrarian"]
            value_buy = value_buy or (np.random.random() < contrarian_boost)
        
        # Apply conviction based on strength of value signal
        if value_buy:
            # Higher conviction for stronger value signals
            conviction_factor = self.personality["conviction"] * (0.8 + 0.4 * value_score)
            conviction_check = np.random.random() < conviction_factor
            if conviction_check:
                return True
        
        # If value analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal with value focus.
        
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
            
        # Calculate value metrics if not already calculated
        if symbol not in self.value_scores:
            self._calculate_value_metrics(market_data, symbol)
            
        # Get value metrics
        value_score = self.value_scores.get(symbol, 0.0)
        is_underpriced = self.underpriced.get(symbol, False)
        
        # Value approach - sell overvalued assets
        weak_value = value_score < (self.undervalue_score_threshold * 0.7)  # Significantly below threshold
        
        # Get price and entry data for this position
        current_price = market_data[symbol].get("price", 0)
        entry_price = self.portfolio[symbol].get("entry_price", current_price)
        
        # Calculate price appreciation
        if entry_price > 0:
            price_appreciation = (current_price - entry_price) / entry_price
        else:
            price_appreciation = 0
            
        # Value sell signal - if no longer undervalued and good profit, or if fundamentals deteriorate
        value_sell = (not is_underpriced and price_appreciation > self.take_profit * 0.7) or weak_value
        
        # Add market sentiment factor - more likely to sell in extreme optimism
        market_sentiment = market_data.get("market_sentiment", 0)  # -1 to 1 scale
        market_euphoria = market_sentiment > 0.5  # High optimism might signal overvaluation
        
        if market_euphoria and not is_underpriced:
            contrarian_factor = self.personality["contrarian"]
            value_sell = value_sell or (np.random.random() < contrarian_factor)
        
        # Apply patience based on holding time
        holding_time = self.portfolio[symbol].get("holding_time", 0)
        min_holding_time = 20  # Arbitrary minimum holding periods
        
        if holding_time < min_holding_time and is_underpriced:
            # Lower probability of selling underpriced assets too early
            patience_check = np.random.random() < (1 - self.personality["patience"])
            if not patience_check:
                return False
                
        # Apply conviction if selling due to value signals
        if value_sell:
            # Higher conviction for clear overvaluation signals
            conviction_factor = self.personality["conviction"] * (0.8 + 0.4 * (1 - value_score))
            conviction_check = np.random.random() < conviction_factor
            if conviction_check:
                return True
        
        # If value analysis doesn't produce a strong signal, default to DQN
        return signal
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a value-investing styled explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        # Get value metrics
        value_score = self.value_scores.get(symbol, 0.0)
        intrinsic_value = self.intrinsic_values.get(symbol, 0)
        is_underpriced = self.underpriced.get(symbol, False)
        
        # Format value description
        value_desc = "fairly valued"
        if value_score > 0.8:
            value_desc = "significantly undervalued"
        elif value_score > 0.6:
            value_desc = "undervalued"
        elif value_score < 0.3:
            value_desc = "overvalued"
        elif value_score < 0.2:
            value_desc = "significantly overvalued"
            
        # Generate explanation based on decision
        if decision == "buy":
            if is_underpriced:
                return f"Value buy signal for {symbol}: Asset appears {value_desc} with satisfactory margin of safety. Focusing on long-term intrinsic value."
            else:
                return f"Buy signal for {symbol}: While not meeting our strict value criteria, this asset shows improving fundamentals and reasonable valuation metrics."
        elif decision == "sell":
            if not is_underpriced:
                return f"Value sell signal for {symbol}: Asset appears {value_desc}, reaching our target price or showing deteriorating fundamentals."
            else:
                return f"Sell signal for {symbol}: Despite favorable valuation, other risk factors or portfolio considerations necessitate liquidation."
        else:
            if is_underpriced:
                return f"Holding {symbol}: Asset remains {value_desc}. Maintaining position as margin of safety persists."
            else:
                return f"Holding {symbol}: Asset is {value_desc}. Monitoring closely for selling opportunity or fundamental changes." 