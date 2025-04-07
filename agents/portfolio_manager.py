"""
Portfolio Manager Agent - Manages portfolio allocation and risk.

This agent is responsible for managing asset allocation, position sizing,
and overall risk management of the trading portfolio.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from models.wallet import Wallet

class PortfolioManagerAgent(BaseAgent):
    """
    Portfolio Manager Agent.
    
    Manages asset allocation, position sizing, and risk across the portfolio.
    Makes allocation decisions based on market conditions, strategy signals,
    and risk parameters.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 config: Dict[str, Any],
                 parent_id: Optional[str] = None,
                 wallet: Optional[Wallet] = None):
        """
        Initialize the portfolio manager agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            config: Configuration parameters
            parent_id: ID of the parent agent
            wallet: Wallet for tracking balances and trades
        """
        super().__init__(
            name=name,
            description=description,
            agent_type="portfolio_manager",
            config=config,
            parent_id=parent_id
        )
        
        # Portfolio parameters
        self.initial_balance = config.get("initial_balance", 10000)
        self.max_allocation_pct = config.get("max_allocation_pct", 20)
        self.max_risk_per_trade_pct = config.get("max_risk_per_trade_pct", 2)
        self.min_cash_reserve_pct = config.get("min_cash_reserve_pct", 15)
        self.rebalance_threshold_pct = config.get("rebalance_threshold_pct", 10)
        self.drawdown_protective_threshold = config.get("drawdown_protective_threshold", 15)
        self.volatility_adjustment = config.get("volatility_adjustment", True)
        self.portfolio_targets = config.get("portfolio_targets", {})
        
        # Initialize wallet for tracking balances and trades
        self.wallet = wallet
        if not self.wallet:
            self.wallet = Wallet(
                initial_balance=self.initial_balance,
                base_currency="USDT",
                name="Portfolio-Wallet"
            )
        
        # Portfolio state
        self.portfolio = {
            "base_currency": config.get("base_currency", "USDT"),
            "total_value": self.initial_balance,
            "cash": self.initial_balance,
            "positions": [],
            "performance": {
                "starting_value": self.initial_balance,
                "peak_value": self.initial_balance,
                "current_drawdown_pct": 0,
                "max_drawdown_pct": 0,
                "total_pnl_pct": 0,
                "daily_returns": [],
            },
            "last_rebalance": datetime.now().timestamp(),
        }
        
        # Position tracking
        self.position_history = []
        
        self.logger.info(f"Portfolio Manager Agent {self.name} initialized with balance: {self.initial_balance}")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data and trading signals to manage portfolio.
        
        Args:
            data: Dictionary containing:
                - market_data: Current market data for all symbols
                - strategy_signals: Trade signals from strategy agents
                - market_analysis: Market analysis results
                - portfolio_state: Current portfolio state (optional)
                
        Returns:
            Updated portfolio state and allocation decisions
        """
        if not data:
            self.logger.warning("No data provided to portfolio manager")
            return {"error": "Missing data for portfolio management"}
            
        market_data = data.get("market_data", {})
        strategy_signals = data.get("strategy_signals", [])
        market_analysis = data.get("market_analysis", {})
        
        # If external portfolio state is provided, update our internal state
        if "portfolio_state" in data:
            self._update_portfolio_state(data["portfolio_state"])
        
        self.logger.info(f"Managing portfolio with {len(strategy_signals)} signals")
        
        try:
            # Update portfolio value based on current market prices
            self._update_portfolio_value(market_data)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Process strategy signals to determine actions
            allocation_decisions = self._process_signals(strategy_signals, market_data, market_analysis)
            
            # Check if rebalancing is needed
            rebalance_actions = []
            if self._should_rebalance():
                rebalance_actions = self._rebalance_portfolio(market_data)
            
            # Implement risk management
            risk_adjustments = self._apply_risk_management(market_data, market_analysis)
            
            result = {
                "portfolio": self.portfolio,
                "allocation_decisions": allocation_decisions,
                "rebalance_actions": rebalance_actions,
                "risk_adjustments": risk_adjustments,
                "timestamp": datetime.now().isoformat()
            }
            
            # Record execution
            self.performance_metrics["total_decisions"] += 1
            self.last_run = datetime.now().timestamp()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio management: {e}", exc_info=True)
            return {"error": f"Portfolio management error: {str(e)}"}
    
    def train(self, training_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Optimize portfolio allocation parameters.
        
        Args:
            training_data: Dictionary containing:
                - historical_market_data: Historical market data
                - historical_signals: Historical strategy signals
                - optimization_target: Target metric to optimize (e.g. "sharpe_ratio")
                
        Returns:
            Tuple of (success, metrics)
        """
        if not training_data or "historical_market_data" not in training_data:
            return False, {"error": "No historical data provided"}
            
        self.logger.info(f"Training Portfolio Manager {self.name}")
        
        try:
            # Define parameter ranges
            param_ranges = {
                "max_allocation_pct": [10, 15, 20, 25, 30],
                "max_risk_per_trade_pct": [1, 1.5, 2, 2.5, 3],
                "min_cash_reserve_pct": [10, 15, 20, 25],
                "drawdown_protective_threshold": [10, 15, 20, 25]
            }
            
            # Placeholder for optimization logic
            # In a real implementation, this would run simulations with different parameters
            
            # Simulate finding best parameters
            best_params = {
                "max_allocation_pct": 20,
                "max_risk_per_trade_pct": 2,
                "min_cash_reserve_pct": 15,
                "drawdown_protective_threshold": 15
            }
            
            best_metric = 1.95  # Simulated Sharpe ratio
            
            # Update parameters if optimization was successful
            if best_params:
                self.update_config(best_params)
                
                # Update internal parameters
                for param, value in best_params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                
                self.logger.info(f"Training completed with best Sharpe ratio: {best_metric}")
                
                return True, {
                    "best_params": best_params,
                    "sharpe_ratio": best_metric
                }
            else:
                return False, {"error": "No improvements found during training"}
                
        except Exception as e:
            self.logger.error(f"Error training Portfolio Manager {self.name}: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def _update_portfolio_state(self, external_state: Dict[str, Any]) -> None:
        """
        Update internal portfolio state from external source.
        
        Args:
            external_state: External portfolio state
        """
        if "total_value" in external_state:
            self.portfolio["total_value"] = external_state["total_value"]
        
        if "cash" in external_state:
            self.portfolio["cash"] = external_state["cash"]
            
        if "positions" in external_state:
            self.portfolio["positions"] = external_state["positions"]
    
    def _update_portfolio_value(self, market_data: Dict[str, Any]) -> None:
        """
        Update portfolio value based on current market prices.
        
        Args:
            market_data: Current market data for all symbols
        """
        total_value = self.portfolio["cash"]
        
        for position in self.portfolio["positions"]:
            symbol = position["symbol"]
            amount = position["amount"]
            
            # Get current price if available
            current_price = None
            if symbol in market_data and not market_data[symbol].empty:
                current_price = market_data[symbol].iloc[-1]["close"]
            
            if current_price is not None:
                position_value = amount * current_price
                position["current_price"] = current_price
                position["current_value"] = position_value
                position["pnl"] = (current_price - position["entry_price"]) * amount
                position["pnl_pct"] = (current_price / position["entry_price"] - 1) * 100
                
                total_value += position_value
            
        # Update total portfolio value
        self.portfolio["total_value"] = total_value
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on current portfolio value."""
        current_value = self.portfolio["total_value"]
        starting_value = self.portfolio["performance"]["starting_value"]
        peak_value = self.portfolio["performance"]["peak_value"]
        
        # Update peak value if needed
        if current_value > peak_value:
            self.portfolio["performance"]["peak_value"] = current_value
            peak_value = current_value
        
        # Calculate current drawdown
        if peak_value > 0:
            current_drawdown = (peak_value - current_value) / peak_value * 100
            self.portfolio["performance"]["current_drawdown_pct"] = current_drawdown
            
            # Update max drawdown if needed
            if current_drawdown > self.portfolio["performance"]["max_drawdown_pct"]:
                self.portfolio["performance"]["max_drawdown_pct"] = current_drawdown
        
        # Calculate total PnL
        if starting_value > 0:
            total_pnl = (current_value / starting_value - 1) * 100
            self.portfolio["performance"]["total_pnl_pct"] = total_pnl
        
        # Add daily return (simplified, in real system would track by date)
        today = datetime.now().date().isoformat()
        daily_returns = self.portfolio["performance"]["daily_returns"]
        
        if not daily_returns or daily_returns[-1]["date"] != today:
            if daily_returns:
                prev_value = daily_returns[-1]["value"]
                daily_return = (current_value / prev_value - 1) * 100
            else:
                daily_return = 0
                
            daily_returns.append({
                "date": today,
                "value": current_value,
                "return": daily_return
            })
            
            # Keep only last 90 days
            if len(daily_returns) > 90:
                self.portfolio["performance"]["daily_returns"] = daily_returns[-90:]
                
        # Update agent performance metrics
        self.performance_metrics["total_profit"] = total_pnl
        self.performance_metrics["max_drawdown"] = self.portfolio["performance"]["max_drawdown_pct"]
    
    def _process_signals(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any], market_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process strategy signals to determine portfolio actions.
        
        Args:
            signals: List of strategy signals
            market_data: Current market data
            market_analysis: Market analysis results
            
        Returns:
            List of allocation decisions
        """
        decisions = []
        
        for signal in signals:
            if "action" not in signal or "symbol" not in signal:
                continue
                
            action = signal["action"]
            symbol = signal["symbol"]
            
            # Skip if we don't have market data for this symbol
            if symbol not in market_data or market_data[symbol].empty:
                continue
                
            # Get current price
            current_price = market_data[symbol].iloc[-1]["close"]
            
            # Process buy signals
            if action == "buy" or action == "strong_buy":
                # Check if we already have a position
                has_position = False
                for position in self.portfolio["positions"]:
                    if position["symbol"] == symbol:
                        has_position = True
                        break
                
                if not has_position:
                    # Calculate position size
                    position_size = self._calculate_position_size(symbol, current_price, signal, market_analysis)
                    
                    # Check if we have enough cash
                    required_cash = position_size * current_price
                    if required_cash <= self.portfolio["cash"]:
                        # Create buy decision
                        decisions.append({
                            "symbol": symbol,
                            "action": "buy",
                            "price": current_price,
                            "amount": position_size,
                            "value": required_cash,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update portfolio (simulating the buy)
                        self._execute_buy(symbol, current_price, position_size, signal.get("confidence", 0.5))
                    
            # Process sell signals
            elif action == "sell" or action == "strong_sell":
                # Find position if we have one
                for i, position in enumerate(self.portfolio["positions"]):
                    if position["symbol"] == symbol:
                        # Create sell decision
                        decisions.append({
                            "symbol": symbol,
                            "action": "sell",
                            "price": current_price,
                            "amount": position["amount"],
                            "value": position["amount"] * current_price,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Update portfolio (simulating the sell)
                        self._execute_sell(symbol, current_price)
                        break
        
        return decisions
    
    def _calculate_position_size(self, symbol: str, price: float, signal: Dict[str, Any], market_analysis: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size based on portfolio constraints and risk.
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal: Trading signal that triggered this calculation
            market_analysis: Market analysis results
            
        Returns:
            Position size to take
        """
        # Get total portfolio value
        portfolio_value = self.portfolio["total_value"]
        
        # Base allocation percentage
        base_allocation_pct = min(self.max_allocation_pct, 100 - self.min_cash_reserve_pct)
        
        # Adjust for signal strength/confidence
        confidence = signal.get("confidence", 0.5)
        adjusted_allocation_pct = base_allocation_pct * confidence
        
        # Adjust for market volatility if enabled
        if self.volatility_adjustment and symbol in market_analysis:
            volatility = market_analysis.get(symbol, {}).get("volatility", {})
            volatility_condition = volatility.get("volatility_condition", "normal")
            
            # Reduce position size in high volatility conditions
            if volatility_condition == "high":
                adjusted_allocation_pct *= 0.7
            elif volatility_condition == "low":
                adjusted_allocation_pct *= 1.2
        
        # Calculate maximum position value
        max_position_value = portfolio_value * (adjusted_allocation_pct / 100)
        
        # Apply risk per trade constraint if stop loss is available
        if "stop_loss" in signal and price > signal["stop_loss"]:
            risk_amount = portfolio_value * (self.max_risk_per_trade_pct / 100)
            risk_per_unit = price - signal["stop_loss"]
            risk_based_size = risk_amount / risk_per_unit
            
            # Use the smaller of the two calculations
            position_size = min(max_position_value / price, risk_based_size)
        else:
            position_size = max_position_value / price
        
        return position_size
    
    def _execute_buy(self, symbol: str, price: float, amount: float, confidence: float) -> None:
        """
        Execute a buy order for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            price: Current price
            amount: Amount to buy in base currency
            confidence: Signal confidence level (0-1)
            
        Updates the positions dictionary and portfolio state.
        """
        # Parse symbol to get the actual asset
        if "/" in symbol:
            base_currency, quote_currency = symbol.split("/")
        else:
            base_currency = symbol
            quote_currency = "USDT"  # Default quote currency
        
        # Calculate quantity based on amount and price
        quantity = amount / price
        
        self.logger.info(f"Executing buy: {quantity} {base_currency} at {price} {quote_currency}")
        
        # Update portfolio via wallet
        try:
            trade = self.wallet.add_trade(
                trade_type="buy",
                from_currency=quote_currency,
                to_currency=base_currency,
                from_amount=amount,
                to_amount=quantity,
                price=price,
                exchange="internal",
                external_id=None
            )
            trade_id = trade["id"]
            
            # Add to positions tracking
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.portfolio["positions"].append({
                "id": position_id,
                "symbol": symbol,
                "entry_price": price,
                "quantity": quantity,
                "value": amount,
                "current_price": price,
                "current_value": amount,
                "entry_time": datetime.now().isoformat(),
                "trade_id": trade_id,
                "pnl": 0,
                "pnl_pct": 0,
                "confidence": confidence
            })
            
            self.logger.info(f"Position opened: {position_id} - {quantity} {base_currency} at {price}")
            
        except ValueError as e:
            self.logger.error(f"Failed to execute buy for {symbol}: {str(e)}")
    
    def _execute_sell(self, symbol: str, price: float) -> None:
        """
        Execute a sell order for all positions of a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            price: Current price
            
        Closes all positions for the symbol and updates trade history.
        """
        # Parse symbol to get the actual asset
        if "/" in symbol:
            base_currency, quote_currency = symbol.split("/")
        else:
            base_currency = symbol
            quote_currency = "USDT"  # Default quote currency
            
        positions_to_close = [position["id"] for position in self.portfolio["positions"] if position["symbol"] == symbol]
        
        if not positions_to_close:
            self.logger.warning(f"No positions found to sell for {symbol}")
            return
        
        for position_id in positions_to_close:
            position = next(position for position in self.portfolio["positions"] if position["id"] == position_id)
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            entry_value = position["value"]
            
            # Calculate exit value and P&L
            exit_value = quantity * price
            pnl = exit_value - entry_value
            pnl_pct = (price / entry_price - 1) * 100
            
            self.logger.info(f"Executing sell: {quantity} {base_currency} at {price} {quote_currency}")
            
            # Update wallet
            try:
                trade = self.wallet.add_trade(
                    trade_type="sell",
                    from_currency=base_currency,
                    to_currency=quote_currency,
                    from_amount=quantity,
                    to_amount=exit_value,
                    price=price,
                    exchange="internal",
                    external_id=None
                )
                
                # Update position data
                position.update({
                    "exit_price": price,
                    "exit_time": datetime.now().isoformat(),
                    "exit_value": exit_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "status": "closed",
                    "holding_period": self._calculate_holding_period(
                        datetime.fromisoformat(position["entry_time"]),
                        datetime.now()
                    ),
                    "exit_trade_id": trade["id"]
                })
                
                # Move to history
                self.position_history.append(position)
                
                # Update trade results for metrics
                if pnl > 0:
                    self.performance_metrics["profitable_trades"] += 1
                
                self.logger.info(f"Position closed: {position_id} with P&L: {pnl:.2f} ({pnl_pct:.2f}%)")
                
            except ValueError as e:
                self.logger.error(f"Failed to execute sell for {symbol}: {str(e)}")
            
            # Remove closed position
            self.portfolio["positions"].remove(position)
    
    def _should_rebalance(self) -> bool:
        """
        Determine if portfolio rebalancing is needed.
        
        Returns:
            True if rebalancing is needed, False otherwise
        """
        # Check time since last rebalance (e.g., at least 7 days)
        days_since_rebalance = (datetime.now().timestamp() - self.portfolio["last_rebalance"]) / (24 * 60 * 60)
        if days_since_rebalance >= 7:
            return True
        
        # Check allocation drift
        if self.portfolio_targets and self.portfolio["positions"]:
            for position in self.portfolio["positions"]:
                symbol = position["symbol"]
                if symbol in self.portfolio_targets:
                    target_pct = self.portfolio_targets[symbol]
                    current_pct = position["current_value"] / self.portfolio["total_value"] * 100
                    
                    # If allocation has drifted beyond threshold
                    if abs(current_pct - target_pct) > self.rebalance_threshold_pct:
                        return True
        
        return False
    
    def _rebalance_portfolio(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rebalance portfolio to target allocations.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of rebalancing actions
        """
        actions = []
        
        # Skip if no targets or empty portfolio
        if not self.portfolio_targets or self.portfolio["total_value"] <= 0:
            return actions
        
        # Calculate current allocations
        current_allocations = {
            "cash": self.portfolio["cash"] / self.portfolio["total_value"] * 100
        }
        
        for position in self.portfolio["positions"]:
            symbol = position["symbol"]
            current_allocations[symbol] = position["current_value"] / self.portfolio["total_value"] * 100
        
        # For each target, determine if adjustment is needed
        for symbol, target_pct in self.portfolio_targets.items():
            current_pct = current_allocations.get(symbol, 0)
            
            # Skip if within threshold
            if abs(current_pct - target_pct) <= self.rebalance_threshold_pct:
                continue
            
            # Skip if we don't have market data
            if symbol not in market_data or market_data[symbol].empty:
                continue
                
            current_price = market_data[symbol].iloc[-1]["close"]
            
            # Calculate target value
            target_value = self.portfolio["total_value"] * (target_pct / 100)
            
            # Find current position if any
            current_position = None
            for position in self.portfolio["positions"]:
                if position["symbol"] == symbol:
                    current_position = position
                    break
            
            current_value = 0 if current_position is None else current_position["current_value"]
            
            # Determine action
            if current_value < target_value:
                # Need to buy more
                additional_value = target_value - current_value
                
                # Check if enough cash
                if additional_value <= self.portfolio["cash"]:
                    additional_amount = additional_value / current_price
                    
                    actions.append({
                        "symbol": symbol,
                        "action": "buy",
                        "reason": "rebalance",
                        "price": current_price,
                        "amount": additional_amount,
                        "value": additional_value
                    })
                    
                    # Update portfolio
                    if current_position:
                        # Add to existing position
                        avg_price = (current_position["entry_price"] * current_position["quantity"] + 
                                     current_price * additional_amount) / (current_position["quantity"] + additional_amount)
                        current_position["quantity"] += additional_amount
                        current_position["entry_price"] = avg_price
                        current_position["current_value"] += additional_value
                    else:
                        # Create new position
                        self._execute_buy(symbol, current_price, additional_amount, 0.5)
                        
            elif current_value > target_value:
                # Need to sell some
                reduction_value = current_value - target_value
                reduction_amount = reduction_value / current_price
                
                # Ensure we're not selling more than we have
                if current_position and reduction_amount < current_position["quantity"]:
                    actions.append({
                        "symbol": symbol,
                        "action": "sell_partial",
                        "reason": "rebalance",
                        "price": current_price,
                        "amount": reduction_amount,
                        "value": reduction_value
                    })
                    
                    # Update portfolio
                    current_position["quantity"] -= reduction_amount
                    current_position["current_value"] -= reduction_value
                    self.portfolio["cash"] += reduction_value
                else:
                    # Sell entire position
                    actions.append({
                        "symbol": symbol,
                        "action": "sell",
                        "reason": "rebalance",
                        "price": current_price,
                        "amount": current_position["quantity"],
                        "value": current_position["current_value"]
                    })
                    
                    self._execute_sell(symbol, current_price)
        
        # Update last rebalance time
        self.portfolio["last_rebalance"] = datetime.now().timestamp()
        
        return actions
    
    def _apply_risk_management(self, market_data: Dict[str, Any], market_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply risk management rules to portfolio.
        
        Args:
            market_data: Current market data
            market_analysis: Market analysis results
            
        Returns:
            List of risk adjustment actions
        """
        actions = []
        
        # Check portfolio drawdown
        current_drawdown = self.portfolio["performance"]["current_drawdown_pct"]
        
        # Defensive measures in case of significant drawdown
        if current_drawdown > self.drawdown_protective_threshold:
            # Reduce exposure by selling weakest positions
            if self.portfolio["positions"]:
                # Sort positions by performance, worst first
                positions_by_performance = sorted(
                    self.portfolio["positions"],
                    key=lambda pos: pos.get("pnl_pct", 0)
                )
                
                # Sell worst performing position(s)
                positions_to_sell = positions_by_performance[:1]  # Just the worst one
                
                for position in positions_to_sell:
                    symbol = position["symbol"]
                    current_price = position["current_price"]
                    
                    actions.append({
                        "symbol": symbol,
                        "action": "sell",
                        "reason": "risk_management_drawdown",
                        "price": current_price,
                        "amount": position["quantity"],
                        "value": position["current_value"]
                    })
                    
                    self._execute_sell(symbol, current_price)
        
        # Check for stop losses
        for position in list(self.portfolio["positions"]):  # Use list() to avoid modification during iteration
            symbol = position["symbol"]
            
            # Skip if we don't have market data
            if symbol not in market_data or market_data[symbol].empty:
                continue
                
            current_price = market_data[symbol].iloc[-1]["close"]
            
            # Check for trailing stop loss (15% drop from peak)
            if "peak_price" not in position:
                position["peak_price"] = position["entry_price"]
            
            if current_price > position["peak_price"]:
                position["peak_price"] = current_price
            
            # Calculate trailing stop loss level (15% below peak)
            trailing_stop = position["peak_price"] * 0.85
            
            # If price is below stop loss, sell
            if current_price < trailing_stop:
                actions.append({
                    "symbol": symbol,
                    "action": "sell",
                    "reason": "trailing_stop_loss",
                    "price": current_price,
                    "amount": position["quantity"],
                    "value": position["current_value"]
                })
                
                self._execute_sell(symbol, current_price)
        
        return actions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio.
        
        Returns:
            Dictionary with portfolio summary
        """
        # Calculate asset allocation
        allocation = {
            "cash": {
                "value": self.portfolio["cash"],
                "percentage": self.portfolio["cash"] / self.portfolio["total_value"] * 100 if self.portfolio["total_value"] > 0 else 0
            },
            "assets": {}
        }
        
        for position in self.portfolio["positions"]:
            symbol = position["symbol"]
            allocation["assets"][symbol] = {
                "value": position["current_value"],
                "percentage": position["current_value"] / self.portfolio["total_value"] * 100 if self.portfolio["total_value"] > 0 else 0,
                "amount": position["quantity"],
                "entry_price": position["entry_price"],
                "current_price": position["current_price"],
                "pnl_pct": position["pnl_pct"]
            }
        
        # Calculate performance metrics
        performance = {
            "total_value": self.portfolio["total_value"],
            "starting_value": self.portfolio["performance"]["starting_value"],
            "total_return_pct": self.portfolio["performance"]["total_pnl_pct"],
            "current_drawdown_pct": self.portfolio["performance"]["current_drawdown_pct"],
            "max_drawdown_pct": self.portfolio["performance"]["max_drawdown_pct"]
        }
        
        # Calculate recent trades
        recent_trades = self.position_history[-10:] if len(self.position_history) > 10 else self.position_history
        
        return {
            "allocation": allocation,
            "performance": performance,
            "recent_trades": recent_trades,
            "timestamp": datetime.now().isoformat()
        } 