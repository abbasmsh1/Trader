"""
Base Trader Agent - Foundation for all trader agents in the system.
"""

import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from agents.base_agent import BaseAgent
from models.wallet import Wallet
from models.database import DatabaseHandler

class BaseTraderAgent(BaseAgent):
    """
    Base trader agent implementation.
    
    This class serves as the foundation for all trader-specific implementations,
    providing common functionality like:
    - Market data processing
    - Position tracking
    - Signal generation
    - Trading decisions
    - Performance metrics
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the base trader agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="trader")
        
        # Initialize database handler
        self.db = DatabaseHandler()
        
        # Extract configuration
        self.trading_style = config.get("trading_style", "conservative")
        self.symbols = config.get("symbols", ["BTC/USDT"])
        self.position_size = config.get("position_size", 0.1)  # 10% of available balance
        self.stop_loss = config.get("stop_loss", 0.05)  # 5% stop loss
        self.take_profit = config.get("take_profit", 0.10)  # 10% take profit
        self.max_positions = config.get("max_positions", 5)  # Maximum number of open positions
        
        # Wallet reference - set later via set_wallet()
        self.wallet = None
        self.personal_wallet = None
        
        # Portfolio tracking
        self.portfolio = {}
        
        # Initialize portfolio tracking for each symbol
        for symbol in self.symbols:
            self.portfolio[symbol] = {
                "position": 0.0,
                "entry_price": 0.0,
                "last_price": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "trades": 0,
                "last_trade_time": None
            }
        
        # Trading metrics
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_sum": 0.0,
            "loss_sum": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "profit_percentage": 0.0
        }
        
        # Decision/signal history
        self.signal_history = []
        self.last_trade_time = None
        self.last_signal = None
        
        # State features
        self.features = {}
        
        # Load any existing state
        self._load_state()
        
        self.logger.info(f"Trader Agent {name} initialized with {len(self.symbols)} symbols")
    
    def _load_state(self) -> None:
        """Load agent state from database."""
        state = self.db.load_agent_state(self.agent_id)
        if state:
            self.portfolio = state.get("portfolio", self.portfolio)
            self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
            self.signal_history = state.get("signal_history", [])
            if "last_trade_time" in state and state["last_trade_time"]:
                self.last_trade_time = datetime.fromisoformat(state["last_trade_time"])
    
    def _save_state(self) -> None:
        """Save agent state to database."""
        state = {
            "portfolio": self.portfolio,
            "performance_metrics": self.performance_metrics,
            "signal_history": self.signal_history[-10:],  # Save only recent signals
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None
        }
        self.db.save_agent_state(self.agent_id, self.__class__.__name__, state)
    
    def initialize(self) -> bool:
        """Initialize the trader agent."""
        try:
            self.logger.info(f"Initializing {self.name} with {len(self.symbols)} symbols")
            
            # Initialize portfolio tracking
            for symbol in self.symbols:
                self.portfolio[symbol] = {
                    "position": 0.0,
                    "entry_price": 0.0,
                    "last_price": 0.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "trades": 0,
                    "last_trade_time": None
                }
            
            # Enforce initial BTC purchase if BTC is in symbols
            if "BTC/USDT" in self.symbols and self.personal_wallet:
                # Allocate 20% of initial balance to BTC
                btc_allocation = self.personal_wallet.initial_balance * 0.2
                btc_price = self.personal_wallet.last_prices.get("BTC", 50000.0)  # Default price if not available
                btc_amount = btc_allocation / btc_price
                
                # Execute initial BTC purchase
                self.personal_wallet.add_trade(
                    trade_type="buy",
                    from_currency="USDT",
                    to_currency="BTC",
                    from_amount=btc_allocation,
                    to_amount=btc_amount,
                    price=btc_price,
                    fee=btc_allocation * 0.001,  # 0.1% fee
                    exchange="initial_allocation"
                )
                
                # Update portfolio tracking
                self.portfolio["BTC/USDT"]["position"] = btc_amount
                self.portfolio["BTC/USDT"]["entry_price"] = btc_price
                self.portfolio["BTC/USDT"]["last_price"] = btc_price
                
                self.logger.info(f"Initial BTC purchase for {self.name}: {btc_amount:.6f} BTC @ {btc_price}")
            
            self.initialized = True
            self.logger.info(f"Trader Agent {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing trader agent: {str(e)}")
            return False
    
    def set_wallet(self, wallet: Wallet) -> None:
        """
        Set the system wallet reference. This is the wallet used by the entire system.
        
        Args:
            wallet: Wallet instance
        """
        self.wallet = wallet
        self.logger.info(f"System wallet set for {self.name}: {wallet.name}")
    
    def set_personal_wallet(self, wallet: Wallet) -> None:
        """
        Set the personal wallet for this trader agent.
        Each trader agent gets its own wallet with personal funds.
        
        Args:
            wallet: Wallet instance for personal use
        """
        self.personal_wallet = wallet
        self.logger.info(f"Personal wallet set for {self.name}: {wallet.name} with {wallet.initial_balance} {wallet.base_currency}")
    
    def _process_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an update cycle.
        
        Args:
            data: Input data for processing, including market data
            
        Returns:
            Results of processing
        """
        # Extract market data
        market_data = data.get("market_data", {})
        
        # If no market data provided, skip update
        if not market_data:
            self.logger.warning("No market data provided for update")
            return {"result": "skipped", "reason": "no_data"}
        
        # Update portfolio with latest prices
        self._update_portfolio(market_data)
        
        # Generate trading decisions
        decisions = self._generate_trading_decisions(market_data)
        
        # Execute trading decisions if enabled
        results = {}
        if self.active:
            results = self._execute_decisions(decisions)
        
        # Update performance metrics
        self._update_metrics()
        
        # Save state
        self._save_state()
        
        return {
            "decisions": decisions,
            "results": results,
            "metrics": self.performance_metrics,
            "portfolio": self.portfolio
        }
    
    def _update_portfolio(self, market_data: Dict[str, Any]) -> None:
        """
        Update portfolio tracking with latest prices.
        
        Args:
            market_data: Market data dictionary with latest prices
        """
        # Extract prices from market data
        prices = {}
        for symbol in self.symbols:
            if symbol in market_data and "price" in market_data[symbol]:
                prices[symbol] = market_data[symbol]["price"]
        
        # If no prices available, skip update
        if not prices:
            self.logger.warning("No price data available in market data")
            return
        
        # Update portfolio for each symbol
        for symbol, price in prices.items():
            if symbol in self.portfolio:
                position = self.portfolio[symbol]["position"]
                entry_price = self.portfolio[symbol]["entry_price"]
                
                # Store last price
                self.portfolio[symbol]["last_price"] = price
                
                # Calculate unrealized PnL if position exists
                if position != 0 and entry_price != 0:
                    if position > 0:  # Long position
                        pnl_pct = (price - entry_price) / entry_price
                    else:  # Short position
                        pnl_pct = (entry_price - price) / entry_price
                    
                    self.portfolio[symbol]["unrealized_pnl"] = pnl_pct
                
                # Save portfolio snapshot
                self.db.save_portfolio(
                    self.agent_id,
                    symbol,
                    position,
                    entry_price,
                    price,
                    self.portfolio[symbol]["unrealized_pnl"],
                    self.portfolio[symbol]["realized_pnl"]
                )
    
    def _generate_trading_decisions(self, market_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate trading decisions for each symbol.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dictionary mapping symbols to decisions (buy, sell, hold)
        """
        decisions = {}
        
        for symbol in self.symbols:
            # Skip if no market data for this symbol
            if symbol not in market_data:
                decisions[symbol] = "hold"
                continue
            
            # Get current position
            position = self.portfolio[symbol]["position"]
            
            # Generate signals
            buy_signal = self._generate_buy_signal(symbol, market_data)
            sell_signal = self._generate_sell_signal(symbol, market_data)
            
            # Make decision
            if position <= 0 and buy_signal:
                decisions[symbol] = "buy"
            elif position > 0 and sell_signal:
                decisions[symbol] = "sell"
            else:
                decisions[symbol] = "hold"
            
            # Create signal data
            signal_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "decision": decisions[symbol],
                "price": market_data[symbol].get("price", 0),
                "strength": self._calculate_signal_strength(symbol, market_data),
                "status": "active"
            }
            
            # Save signal to database
            self.db.add_signal(self.id, signal_data)
            
            # Record the signal in agent's history
            self.last_signal = signal_data
            self.signal_history.append(self.last_signal)
            
            # Trim signal history if too long
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
        
        return decisions
    
    def _generate_buy_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate buy signal for the given symbol.
        To be overridden by subclasses for specific strategies.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to buy
        """
        # Base implementation - random decision with 5% probability
        return np.random.rand() < 0.05
    
    def _generate_sell_signal(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Generate sell signal for the given symbol.
        To be overridden by subclasses for specific strategies.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating whether to sell
        """
        # Base implementation - random decision with 5% probability
        return np.random.rand() < 0.05
    
    def _calculate_signal_strength(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """
        Calculate the strength of the trading signal.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Signal strength between 0 and 1
        """
        # Base implementation - return random strength
        return np.random.random()
    
    def _execute_decisions(self, decisions: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute trading decisions.
        
        Args:
            decisions: Dictionary mapping symbols to decisions
            
        Returns:
            Results of execution
        """
        results = {}
        
        for symbol, decision in decisions.items():
            results[symbol] = {"action": decision, "executed": False}
            
            # Skip if decision is to hold
            if decision == "hold":
                continue
            
            # Skip if no personal wallet set
            if not self.personal_wallet:
                self.logger.warning(f"Cannot execute {decision} for {symbol}: No personal wallet set")
                results[symbol]["reason"] = "no_wallet"
                continue
            
            # Skip if no market data available
            if symbol not in self.portfolio or "last_price" not in self.portfolio[symbol]:
                self.logger.warning(f"Cannot execute {decision} for {symbol}: No price data available")
                results[symbol]["reason"] = "no_price"
                continue
            
            # Get price and position info
            price = self.portfolio[symbol]["last_price"]
            position = self.portfolio[symbol]["position"]
            
            try:
                if decision == "buy":
                    # Calculate position size
                    base_currency = self.personal_wallet.base_currency
                    available_balance = self.personal_wallet.get_balance(base_currency)
                    
                    # Calculate trade amount
                    trade_amount = available_balance * self.position_size
                    
                    # Check if trade amount meets minimum requirement
                    min_trade_value = self.config.get("min_trade_value", 5.0)
                    if trade_amount < min_trade_value:
                        self.logger.info(f"Buy amount too small for {symbol}: {trade_amount:.2f} {base_currency} < {min_trade_value:.2f} {base_currency}")
                        results[symbol]["reason"] = "amount_too_small"
                        continue
                    
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
                    
                    # Save trade record
                    self.db.save_trade(
                        self.agent_id,
                        symbol,
                        "buy",
                        quantity,
                        price,
                        trade_amount,
                        trade_amount * 0.001,
                        {"wallet_id": self.personal_wallet.wallet_id}
                    )
                    
                    # Update portfolio tracking
                    self._update_portfolio_after_trade(symbol, "buy", quantity, price)
                    
                    # Mark as executed
                    results[symbol]["executed"] = True
                    results[symbol]["details"] = {
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
                        results[symbol]["reason"] = "amount_too_small"
                        continue
                    
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
                    
                    # Save trade record
                    self.db.save_trade(
                        self.agent_id,
                        symbol,
                        "sell",
                        quantity,
                        price,
                        trade_amount,
                        trade_amount * 0.001,
                        {"wallet_id": self.personal_wallet.wallet_id}
                    )
                    
                    # Update portfolio tracking
                    self._update_portfolio_after_trade(symbol, "sell", quantity, price)
                    
                    # Mark as executed
                    results[symbol]["executed"] = True
                    results[symbol]["details"] = {
                        "price": price,
                        "quantity": quantity,
                        "value": trade_amount
                    }
                    
                    self.logger.info(f"Executed SELL for {symbol}: {quantity:.6f} @ {price:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error executing {decision} for {symbol}: {str(e)}")
                results[symbol]["error"] = str(e)
        
        return results
    
    def _update_portfolio_after_trade(self, symbol: str, trade_type: str, quantity: float, price: float) -> None:
        """
        Update portfolio tracking after a trade.
        
        Args:
            symbol: Trading symbol
            trade_type: Type of trade (buy or sell)
            quantity: Quantity traded
            price: Trade price
        """
        # Update last trade time
        current_time = datetime.now()
        self.last_trade_time = current_time
        self.portfolio[symbol]["last_trade_time"] = current_time
        
        # Update trade count
        self.portfolio[symbol]["trades"] += 1
        self.performance_metrics["total_trades"] += 1
        
        # Update position and PnL
        if trade_type == "buy":
            # Update position and entry price
            self.portfolio[symbol]["position"] += quantity
            self.portfolio[symbol]["entry_price"] = price
            
        elif trade_type == "sell":
            # Calculate realized PnL
            entry_price = self.portfolio[symbol]["entry_price"]
            pnl_pct = (price - entry_price) / entry_price
            realized_pnl = pnl_pct
            
            # Update realized PnL
            self.portfolio[symbol]["realized_pnl"] += realized_pnl
            
            # Update performance metrics
            if realized_pnl > 0:
                self.performance_metrics["winning_trades"] += 1
                self.performance_metrics["profit_sum"] += realized_pnl
                self.performance_metrics["largest_win"] = max(realized_pnl, self.performance_metrics["largest_win"])
            else:
                self.performance_metrics["losing_trades"] += 1
                self.performance_metrics["loss_sum"] += abs(realized_pnl)
                self.performance_metrics["largest_loss"] = max(abs(realized_pnl), self.performance_metrics["largest_loss"])
            
            # Reset position
            self.portfolio[symbol]["position"] = 0
            self.portfolio[symbol]["entry_price"] = 0
            self.portfolio[symbol]["unrealized_pnl"] = 0
        
        # Save performance metrics
        self.db.save_performance_metrics(self.agent_id, self.performance_metrics)
    
    def _update_metrics(self) -> None:
        """Update performance metrics."""
        # Calculate win rate
        total_trades = self.performance_metrics["total_trades"]
        if total_trades > 0:
            self.performance_metrics["win_rate"] = (self.performance_metrics["winning_trades"] / total_trades) * 100
        
        # Calculate average win/loss
        winning_trades = self.performance_metrics["winning_trades"]
        if winning_trades > 0:
            self.performance_metrics["average_win"] = self.performance_metrics["profit_sum"] / winning_trades
        
        losing_trades = self.performance_metrics["losing_trades"]
        if losing_trades > 0:
            self.performance_metrics["average_loss"] = self.performance_metrics["loss_sum"] / losing_trades
        
        # Calculate profit factor
        if self.performance_metrics["loss_sum"] > 0:
            self.performance_metrics["profit_factor"] = self.performance_metrics["profit_sum"] / self.performance_metrics["loss_sum"]
        
        # Calculate overall profit percentage
        if self.personal_wallet:
            initial_balance = self.personal_wallet.initial_balance
            current_value = self.personal_wallet.get_total_value()
            if initial_balance > 0:
                self.performance_metrics["profit_percentage"] = ((current_value - initial_balance) / initial_balance) * 100
        
        # Save performance metrics
        self.db.save_performance_metrics(self.agent_id, self.performance_metrics)
    
    def _calculate_reward(self, state: Dict[str, Any], action: str, next_state: Dict[str, Any]) -> float:
        """
        Calculate reward for reinforcement learning.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Reward value
        """
        # Simple reward based on PnL
        if "portfolio_value" in state and "portfolio_value" in next_state:
            return next_state["portfolio_value"] - state["portfolio_value"]
        
        return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent for persistence.
        
        Returns:
            Dictionary representing agent state
        """
        # Get base state from parent class
        state = super().get_state()
        
        # Add trader-specific state
        state["portfolio"] = self.portfolio
        state["performance_metrics"] = self.performance_metrics
        state["last_trade_time"] = self.last_trade_time.isoformat() if self.last_trade_time else None
        state["signal_history"] = self.signal_history[-10:]  # Save only recent signals
        
        return state
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load trader agent state from persisted data.
        
        Args:
            state: State dictionary to load
            
        Returns:
            Boolean indicating success
        """
        # Load base state from parent class
        success = super().load_state(state)
        if not success:
            return False
        
        try:
            # Load trader-specific state
            if "portfolio" in state:
                self.portfolio = state["portfolio"]
            
            if "performance_metrics" in state:
                self.performance_metrics = state["performance_metrics"]
            
            if "last_trade_time" in state and state["last_trade_time"]:
                self.last_trade_time = datetime.fromisoformat(state["last_trade_time"])
            
            if "signal_history" in state:
                self.signal_history = state["signal_history"]
            
            self.logger.info(f"Loaded trader state for {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading trader state: {str(e)}")
            return False
    
    def explain_decision(self, symbol: str, decision: str) -> str:
        """
        Generate a natural language explanation for a trading decision.
        
        Args:
            symbol: Trading symbol
            decision: Trading decision (buy, sell, hold)
            
        Returns:
            Explanation string
        """
        # Base implementation - can be overridden by subclasses for more specific explanations
        if decision == "buy":
            return f"I decided to buy {symbol} based on my trading strategy and market conditions."
        elif decision == "sell":
            return f"I decided to sell {symbol} based on my trading strategy and market conditions."
        else:
            return f"I decided to hold {symbol} for now, as the conditions for trading are not met."
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history for the agent.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of trade records
        """
        return self.db.get_trade_history(self.agent_id, limit)
    
    def get_portfolio_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get portfolio history for the agent.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of records to return
            
        Returns:
            List of portfolio records
        """
        return self.db.get_portfolio_history(self.agent_id, symbol, limit)
    
    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get performance metrics history for the agent.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of performance metrics records
        """
        return self.db.get_performance_history(self.agent_id, limit) 