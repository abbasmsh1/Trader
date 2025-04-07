"""
Swing Strategy Agent - Trades market reversals and oscillations.

This agent implements swing trading strategies based on overbought/oversold conditions
and price reversals at support and resistance levels.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from agents.base_agent import BaseAgent

class SwingStrategyAgent(BaseAgent):
    """
    Swing Strategy Agent that identifies and trades market reversals.
    
    This agent focuses on identifying overbought and oversold conditions
    using oscillator indicators and trading price reversals at key levels.
    """
    
    def __init__(self, agent_id: str, name: str, description: str, config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the swing strategy agent."""
        super().__init__(agent_id, name, description, "swing_strategy", config, parent_id)
        
        # Strategy parameters
        self.oversold_threshold = self.config.get("oversold_threshold", 30)
        self.overbought_threshold = self.config.get("overbought_threshold", 70)
        self.reversion_strength = self.config.get("reversion_strength", 0.05)
        self.min_holding_period = self.config.get("min_holding_period", 12)  # hours
        self.max_holding_period = self.config.get("max_holding_period", 96)  # hours
        self.position_size_pct = self.config.get("position_size_pct", 8)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 8)
        self.profit_target_pct = self.config.get("profit_target_pct", 12)
        self.max_trades = self.config.get("max_trades", 3)
        
        # Active trades tracking
        self.active_trades = {}
        self.signals_history = []
        self.performance_history = []
        
        self.logger = logging.getLogger(f"swing_strategy_{agent_id}")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the swing trading strategy on market data.
        
        Args:
            data: Dictionary containing market data and portfolio state
                 Expected format:
                 {
                     "market_data": {
                         "BTC/USDT": {
                             "ohlcv": [...],  # List of OHLCV candles
                             "indicators": {
                                 "rsi": [...],
                                 "stoch": {...},
                                 "bbands": {...},
                                 ...
                             }
                         },
                         ...
                     },
                     "portfolio": {...}
                 }
        
        Returns:
            Dictionary with trade signals and strategy state
        """
        if not self.is_active:
            return {"error": "Agent is not active"}
        
        try:
            self.logger.info("Running swing strategy")
            
            market_data = data.get("market_data", {})
            portfolio = data.get("portfolio", {})
            
            # Update active trades with latest prices
            self._update_active_trades(market_data)
            
            # Generate new signals
            signals = []
            for symbol, data in market_data.items():
                # Skip if we don't have the necessary data
                if not self._has_required_data(data):
                    continue
                
                # Check for oversold/overbought conditions
                signal = self._check_reversal_conditions(symbol, data)
                
                if signal:
                    # Verify we're not exceeding max trades
                    if signal["side"] == "buy" and len([t for t in self.active_trades.values() if t["side"] == "buy"]) >= self.max_trades:
                        self.logger.info(f"Skipping buy signal for {symbol}: max trades reached")
                        continue
                        
                    signals.append(signal)
            
            # Record signals to history
            for signal in signals:
                self.signals_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": signal["symbol"],
                    "side": signal["side"],
                    "price": signal["price"],
                    "strength": signal["strength"],
                    "reasoning": signal["reasoning"]
                })
            
            # Trim history if it's getting too long
            if len(self.signals_history) > 100:
                self.signals_history = self.signals_history[-100:]
            
            return {
                "signals": signals,
                "active_trades": list(self.active_trades.values()),
                "signals_history": self.signals_history[-5:],  # Return only recent signals
                "performance": self._get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error in swing strategy run: {str(e)}")
            return {
                "error": str(e),
                "signals": [],
                "active_trades": list(self.active_trades.values())
            }
    
    def _has_required_data(self, market_data: Dict[str, Any]) -> bool:
        """Check if we have all required data for this strategy."""
        indicators = market_data.get("indicators", {})
        return all(ind in indicators for ind in ["rsi", "stoch", "bbands"])
    
    def _check_reversal_conditions(self, symbol: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check for reversal conditions based on overbought/oversold indicators.
        
        Returns a signal dict if a valid signal is found, otherwise None.
        """
        indicators = data.get("indicators", {})
        ohlcv = data.get("ohlcv", [])
        
        if not indicators or not ohlcv or len(ohlcv) < 5:
            return None
            
        # Get latest price
        latest_price = ohlcv[-1][4]  # Close price of last candle
        
        # Extract indicator values
        rsi = indicators.get("rsi", [])
        stoch = indicators.get("stoch", {})
        bbands = indicators.get("bbands", {})
        
        if not rsi or not stoch or not bbands:
            return None
            
        # Get latest values
        current_rsi = rsi[-1]
        current_stoch_k = stoch.get("k", [])[-1] if stoch.get("k") else 50
        current_stoch_d = stoch.get("d", [])[-1] if stoch.get("d") else 50
        
        bb_upper = bbands.get("upper", [])[-1] if bbands.get("upper") else latest_price * 1.05
        bb_lower = bbands.get("lower", [])[-1] if bbands.get("lower") else latest_price * 0.95
        bb_width = (bb_upper - bb_lower) / ((bb_upper + bb_lower) / 2)
        
        # Calculate price distance from Bollinger Bands
        price_to_upper = (bb_upper - latest_price) / latest_price
        price_to_lower = (latest_price - bb_lower) / latest_price
        
        # Check for oversold conditions (buy signal)
        buy_signal_strength = 0
        buy_reasons = []
        
        if current_rsi < self.oversold_threshold:
            buy_signal_strength += 1
            buy_reasons.append(f"RSI oversold ({current_rsi:.1f})")
            
        if current_stoch_k < self.oversold_threshold and current_stoch_d < self.oversold_threshold:
            buy_signal_strength += 1
            buy_reasons.append(f"Stochastic oversold (K:{current_stoch_k:.1f}, D:{current_stoch_d:.1f})")
            
        if price_to_lower < 0.005:  # Price close to lower band
            buy_signal_strength += 1
            buy_reasons.append(f"Price near lower Bollinger Band")
            
        # Check for overbought conditions (sell signal)
        sell_signal_strength = 0
        sell_reasons = []
        
        if current_rsi > self.overbought_threshold:
            sell_signal_strength += 1
            sell_reasons.append(f"RSI overbought ({current_rsi:.1f})")
            
        if current_stoch_k > self.overbought_threshold and current_stoch_d > self.overbought_threshold:
            sell_signal_strength += 1
            sell_reasons.append(f"Stochastic overbought (K:{current_stoch_k:.1f}, D:{current_stoch_d:.1f})")
            
        if price_to_upper < 0.005:  # Price close to upper band
            sell_signal_strength += 1
            sell_reasons.append(f"Price near upper Bollinger Band")
        
        # Check for crossovers and divergences (more advanced signals)
        # [Implementation omitted for brevity]
        
        # Determine if we have a valid signal
        if buy_signal_strength >= 2 and buy_signal_strength > sell_signal_strength:
            # Check if this symbol already has an active buy trade
            if any(t["symbol"] == symbol and t["side"] == "buy" for t in self.active_trades.values()):
                return None
                
            # Calculate position size, stop loss and take profit
            stop_loss = latest_price * (1 - self.stop_loss_pct / 100)
            take_profit = latest_price * (1 + self.profit_target_pct / 100)
            
            return {
                "symbol": symbol,
                "side": "buy",
                "price": latest_price,
                "strength": buy_signal_strength,
                "reasoning": ", ".join(buy_reasons),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size_pct": self.position_size_pct,
                "timestamp": datetime.now().isoformat(),
                "id": f"swing_buy_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
            
        elif sell_signal_strength >= 2 and sell_signal_strength > buy_signal_strength:
            # For this example, we'll only generate sell signals for positions we hold
            active_buy = None
            for trade_id, trade in self.active_trades.items():
                if trade["symbol"] == symbol and trade["side"] == "buy":
                    active_buy = trade_id
                    break
            
            if not active_buy:
                return None
                
            return {
                "symbol": symbol,
                "side": "sell",
                "price": latest_price,
                "strength": sell_signal_strength,
                "reasoning": ", ".join(sell_reasons),
                "reference_trade_id": active_buy,
                "timestamp": datetime.now().isoformat(),
                "id": f"swing_sell_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            }
        
        return None
    
    def _update_active_trades(self, market_data: Dict[str, Any]) -> None:
        """Update active trades with latest prices and check for exits."""
        trades_to_remove = []
        
        for trade_id, trade in self.active_trades.items():
            symbol = trade["symbol"]
            
            # Skip if we don't have market data for this symbol
            if symbol not in market_data:
                continue
                
            # Get latest price
            latest_price = market_data[symbol]["ohlcv"][-1][4]
            
            # Update current price
            trade["current_price"] = latest_price
            trade["pnl_pct"] = ((latest_price / trade["entry_price"]) - 1) * 100 if trade["side"] == "buy" else ((trade["entry_price"] / latest_price) - 1) * 100
            
            # Check stop loss and take profit for buy trades
            if trade["side"] == "buy":
                # Check stop loss
                if latest_price <= trade["stop_loss"]:
                    trade["status"] = "closed"
                    trade["exit_price"] = latest_price
                    trade["exit_reason"] = "stop_loss"
                    trade["exit_time"] = datetime.now().isoformat()
                    
                    self._record_trade_performance(trade)
                    trades_to_remove.append(trade_id)
                    
                # Check take profit
                elif latest_price >= trade["take_profit"]:
                    trade["status"] = "closed"
                    trade["exit_price"] = latest_price
                    trade["exit_reason"] = "take_profit"
                    trade["exit_time"] = datetime.now().isoformat()
                    
                    self._record_trade_performance(trade)
                    trades_to_remove.append(trade_id)
                    
                # Check max holding period
                elif "entry_time" in trade:
                    entry_time = datetime.fromisoformat(trade["entry_time"])
                    now = datetime.now()
                    hours_held = (now - entry_time).total_seconds() / 3600
                    
                    if hours_held > self.max_holding_period:
                        trade["status"] = "closed"
                        trade["exit_price"] = latest_price
                        trade["exit_reason"] = "max_holding_period"
                        trade["exit_time"] = datetime.now().isoformat()
                        
                        self._record_trade_performance(trade)
                        trades_to_remove.append(trade_id)
        
        # Remove closed trades
        for trade_id in trades_to_remove:
            del self.active_trades[trade_id]
    
    def _record_trade_performance(self, trade: Dict[str, Any]) -> None:
        """Record closed trade performance metrics."""
        if trade["side"] == "buy":
            pnl_pct = ((trade["exit_price"] / trade["entry_price"]) - 1) * 100
        else:
            pnl_pct = ((trade["entry_price"] / trade["exit_price"]) - 1) * 100
            
        trade_record = {
            "id": trade["id"],
            "symbol": trade["symbol"],
            "side": trade["side"],
            "entry_price": trade["entry_price"],
            "exit_price": trade["exit_price"],
            "entry_time": trade["entry_time"],
            "exit_time": trade["exit_time"],
            "exit_reason": trade["exit_reason"],
            "pnl_pct": pnl_pct,
            "holding_period_hours": self._calculate_holding_hours(trade["entry_time"], trade["exit_time"])
        }
        
        self.performance_history.append(trade_record)
        
        # Trim history if too long
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            
        self.logger.info(f"Trade closed: {trade['symbol']} {trade['side']} - PnL: {pnl_pct:.2f}% - Reason: {trade['exit_reason']}")
    
    def _calculate_holding_hours(self, entry_time_str: str, exit_time_str: str) -> float:
        """Calculate how many hours a position was held."""
        entry_time = datetime.fromisoformat(entry_time_str)
        exit_time = datetime.fromisoformat(exit_time_str)
        return (exit_time - entry_time).total_seconds() / 3600
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return strategy performance metrics."""
        if not self.performance_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "average_profit": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "average_holding_period": 0
            }
            
        # Calculate metrics
        total_trades = len(self.performance_history)
        winning_trades = [t for t in self.performance_history if t["pnl_pct"] > 0]
        losing_trades = [t for t in self.performance_history if t["pnl_pct"] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = np.mean([t["pnl_pct"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t["pnl_pct"] for t in winning_trades)
        total_loss = abs(sum(t["pnl_pct"] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else total_profit
        
        avg_holding = np.mean([t["holding_period_hours"] for t in self.performance_history])
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate * 100,  # as percentage
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "profit_factor": profit_factor,
            "average_holding_period": avg_holding
        }
    
    def process_signal(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trading signal, potentially creating a new trade.
        
        Args:
            signal: Signal details
            portfolio: Current portfolio state
            
        Returns:
            Dictionary with processing result
        """
        try:
            symbol = signal["symbol"]
            side = signal["side"]
            price = signal["price"]
            
            if side == "buy":
                # Create new trade
                trade_id = signal["id"]
                
                self.active_trades[trade_id] = {
                    "id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "entry_price": price,
                    "current_price": price,
                    "stop_loss": signal["stop_loss"],
                    "take_profit": signal["take_profit"],
                    "position_size_pct": signal["position_size_pct"],
                    "status": "open",
                    "entry_time": datetime.now().isoformat(),
                    "pnl_pct": 0.0
                }
                
                return {
                    "success": True,
                    "trade_id": trade_id,
                    "action": "opened",
                    "message": f"Opened {side} position for {symbol}"
                }
                
            elif side == "sell":
                # Close existing trade
                reference_id = signal.get("reference_trade_id")
                
                if reference_id and reference_id in self.active_trades:
                    trade = self.active_trades[reference_id]
                    
                    trade["status"] = "closed"
                    trade["exit_price"] = price
                    trade["exit_reason"] = "signal"
                    trade["exit_time"] = datetime.now().isoformat()
                    
                    self._record_trade_performance(trade)
                    del self.active_trades[reference_id]
                    
                    return {
                        "success": True,
                        "trade_id": reference_id,
                        "action": "closed",
                        "message": f"Closed {trade['side']} position for {symbol}"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"No matching open trade found for {symbol}"
                    }
            
            return {
                "success": False,
                "error": f"Unrecognized signal side: {side}"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the strategy using historical market data.
        
        Args:
            training_data: Historical market data for training
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training swing strategy with historical data")
        
        try:
            # Extract historical data
            historical_data = training_data.get("historical_data", {})
            if not historical_data:
                return {
                    "success": False,
                    "error": "No historical data provided for training"
                }
            
            # Track performance metrics during optimization
            best_params = {}
            best_profit = 0
            best_metrics = {}
            
            # Simplified parameter optimization
            for oversold in [20, 25, 30]:
                for overbought in [70, 75, 80]:
                    for stop_loss in [5, 7, 10]:
                        for profit_target in [8, 12, 15]:
                            params = {
                                "oversold_threshold": oversold,
                                "overbought_threshold": overbought,
                                "stop_loss_pct": stop_loss,
                                "profit_target_pct": profit_target
                            }
                            
                            metrics = self._backtest_parameters(historical_data, params)
                            
                            # Track best performing parameters
                            total_profit = metrics.get("total_profit", 0)
                            if total_profit > best_profit:
                                best_profit = total_profit
                                best_params = params
                                best_metrics = metrics
            
            # Update parameters if we found better ones
            if best_params:
                old_params = {
                    "oversold_threshold": self.oversold_threshold,
                    "overbought_threshold": self.overbought_threshold,
                    "stop_loss_pct": self.stop_loss_pct,
                    "profit_target_pct": self.profit_target_pct
                }
                
                self.oversold_threshold = best_params["oversold_threshold"]
                self.overbought_threshold = best_params["overbought_threshold"]
                self.stop_loss_pct = best_params["stop_loss_pct"]
                self.profit_target_pct = best_params["profit_target_pct"]
                
                # Update config
                for key, value in best_params.items():
                    self.config[key] = value
                
                return {
                    "success": True,
                    "old_params": old_params,
                    "new_params": best_params,
                    "improvement_pct": (best_profit / max(1, best_metrics.get("initial_profit", 1)) - 1) * 100,
                    "metrics": best_metrics
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to find improved parameters"
                }
            
        except Exception as e:
            self.logger.error(f"Error training swing strategy: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _backtest_parameters(self, historical_data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest a set of parameters on historical data.
        
        This is a simplified backtest implementation.
        
        Args:
            historical_data: Historical market data
            params: Parameters to test
            
        Returns:
            Dictionary with backtest metrics
        """
        # This would be a more complex implementation in practice
        # For now, just simulate some results
        
        # Simulated metric calculation based on parameters
        oversold = params["oversold_threshold"]
        overbought = params["overbought_threshold"]
        stop_loss = params["stop_loss_pct"]
        profit_target = params["profit_target_pct"]
        
        # The more extreme the oversold/overbought values, the fewer trades but higher accuracy
        trade_frequency = (40 - oversold) + (overbought - 60)  # 0-40 scale
        
        # Base win rate on the difference between overbought/oversold thresholds
        base_win_rate = 50 + (overbought - oversold) / 5  # 50-60 range
        
        # Risk/reward ratio
        risk_reward = profit_target / stop_loss
        
        # Calculate expected profitability
        expected_win_rate = min(70, base_win_rate + (risk_reward - 1) * 5)
        expected_profit_per_trade = (expected_win_rate / 100 * profit_target) - ((100 - expected_win_rate) / 100 * stop_loss)
        
        # Simulate total profit
        simulated_trades = trade_frequency * 10  # scale factor
        total_profit = expected_profit_per_trade * simulated_trades
        
        # Return metrics
        return {
            "total_profit": total_profit,
            "win_rate": expected_win_rate,
            "trade_count": simulated_trades,
            "profit_per_trade": expected_profit_per_trade,
            "risk_reward_ratio": risk_reward,
            "initial_profit": 20  # baseline for comparison
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the strategy.
        
        Returns:
            Dictionary with state data to be persisted
        """
        return {
            "active_trades": self.active_trades,
            "signals_history": self.signals_history,
            "performance_history": self.performance_history,
            "parameters": {
                "oversold_threshold": self.oversold_threshold,
                "overbought_threshold": self.overbought_threshold,
                "reversion_strength": self.reversion_strength,
                "stop_loss_pct": self.stop_loss_pct,
                "profit_target_pct": self.profit_target_pct
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load a previously saved state.
        
        Args:
            state: Previously saved state data
            
        Returns:
            Boolean indicating success
        """
        try:
            if "active_trades" in state:
                self.active_trades = state["active_trades"]
                
            if "signals_history" in state:
                self.signals_history = state["signals_history"]
                
            if "performance_history" in state:
                self.performance_history = state["performance_history"]
                
            if "parameters" in state:
                params = state["parameters"]
                for param, value in params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                        self.config[param] = value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading swing strategy state: {str(e)}")
            return False 