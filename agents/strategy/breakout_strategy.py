"""
Breakout Strategy Agent - Trades price breakouts from ranges and patterns.

This agent identifies consolidation patterns and trades the breakouts with
momentum when price moves outside established ranges on increased volume.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from agents.base_agent import BaseAgent

class BreakoutStrategyAgent(BaseAgent):
    """
    Breakout Strategy Agent that identifies and trades price breakouts.
    
    This agent looks for price consolidation followed by strong directional
    movement with volume confirmation, and enters trades in the direction
    of the breakout.
    """
    
    def __init__(self, agent_id: str, name: str, description: str, config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the breakout strategy agent."""
        super().__init__(agent_id, name, description, "breakout_strategy", config, parent_id)
        
        # Strategy parameters
        self.breakout_period = self.config.get("breakout_period", 20)
        self.volume_multiplier_threshold = self.config.get("volume_multiplier_threshold", 2.0)
        self.confirmation_candles = self.config.get("confirmation_candles", 2)
        self.false_breakout_filter = self.config.get("false_breakout_filter", True)
        self.position_size_pct = self.config.get("position_size_pct", 5)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 7)
        self.profit_target_pct = self.config.get("profit_target_pct", 20)
        self.max_trades = self.config.get("max_trades", 3)
        
        # Active trades tracking
        self.active_trades = {}
        self.signals_history = []
        self.performance_history = []
        
        # Breakout pattern tracking
        self.ranges = {}  # symbol -> range data
        
        self.logger = logging.getLogger(f"breakout_strategy_{agent_id}")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the breakout trading strategy on market data.
        
        Args:
            data: Dictionary containing market data and portfolio state
                 Expected format:
                 {
                     "market_data": {
                         "BTC/USDT": {
                             "ohlcv": [...],  # List of OHLCV candles
                             "indicators": {...}
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
            self.logger.info("Running breakout strategy")
            
            market_data = data.get("market_data", {})
            portfolio = data.get("portfolio", {})
            
            # Update active trades with latest prices
            self._update_active_trades(market_data)
            
            # Update range data for each symbol
            for symbol, data in market_data.items():
                ohlcv = data.get("ohlcv", [])
                if len(ohlcv) >= self.breakout_period:
                    self._update_range_data(symbol, ohlcv)
            
            # Generate new signals
            signals = []
            for symbol, data in market_data.items():
                ohlcv = data.get("ohlcv", [])
                if len(ohlcv) < self.breakout_period:
                    continue
                    
                # Check for breakouts
                signal = self._check_breakout(symbol, ohlcv)
                
                if signal:
                    # Verify we're not exceeding max trades
                    if len(self.active_trades) >= self.max_trades:
                        self.logger.info(f"Skipping signal for {symbol}: max trades reached")
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
                "signals_history": self.signals_history[-5:],
                "ranges": self.ranges,
                "performance": self._get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error in breakout strategy run: {str(e)}")
            return {
                "error": str(e),
                "signals": [],
                "active_trades": list(self.active_trades.values())
            }
    
    def _update_range_data(self, symbol: str, ohlcv: List[List[float]]) -> None:
        """
        Update the consolidation range data for a symbol.
        
        Args:
            symbol: Trading symbol
            ohlcv: OHLCV candle data
        """
        # Extract recent candles for the range calculation
        recent_candles = ohlcv[-self.breakout_period:]
        
        # Extract highs and lows
        highs = [candle[2] for candle in recent_candles]
        lows = [candle[3] for candle in recent_candles]
        
        # Calculate range levels
        range_high = max(highs[:-1])  # Exclude the most recent candle
        range_low = min(lows[:-1])  # Exclude the most recent candle
        
        # Calculate average volume
        volumes = [candle[5] for candle in recent_candles[:-1]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        # Calculate volatility
        closes = [candle[4] for candle in recent_candles]
        price_changes = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = sum(price_changes) / len(price_changes) if price_changes else 0
        
        # Determine if this is a tight range
        range_size = (range_high - range_low) / range_low
        is_tight_range = range_size < 0.05  # 5% range is considered tight
        
        # Update range data
        self.ranges[symbol] = {
            "high": range_high,
            "low": range_low,
            "mid": (range_high + range_low) / 2,
            "size": range_size * 100,  # As percentage
            "avg_volume": avg_volume,
            "volatility": volatility * 100,  # As percentage
            "is_tight_range": is_tight_range,
            "updated_at": datetime.now().isoformat()
        }
    
    def _check_breakout(self, symbol: str, ohlcv: List[List[float]]) -> Optional[Dict[str, Any]]:
        """
        Check if a price breakout has occurred.
        
        Args:
            symbol: Trading symbol
            ohlcv: OHLCV data
            
        Returns:
            Signal dictionary if a breakout is detected, otherwise None
        """
        if symbol not in self.ranges:
            return None
            
        range_data = self.ranges[symbol]
        range_high = range_data["high"]
        range_low = range_data["low"]
        avg_volume = range_data["avg_volume"]
        
        # Get most recent candles for confirmation
        confirm_candles = ohlcv[-(self.confirmation_candles+1):]
        
        # Latest price and volume
        latest_close = confirm_candles[-1][4]
        latest_high = confirm_candles[-1][2]
        latest_low = confirm_candles[-1][3]
        latest_volume = confirm_candles[-1][5]
        
        # Check if we have a volume surge
        volume_surge = latest_volume > (avg_volume * self.volume_multiplier_threshold)
        
        # Strength factors
        strength_factors = []
        
        # Check for bullish breakout (price breaks above range high)
        if latest_close > range_high:
            # Confirm breakout with previous candles if required
            confirmed = True
            
            if self.confirmation_candles > 1:
                # Check if previous candles were inside the range
                for i in range(1, self.confirmation_candles):
                    prev_candle = confirm_candles[-(i+1)]
                    if prev_candle[4] > range_high:  # Previous close was also above
                        confirmed = False
                        break
            
            # Apply false breakout filter if enabled
            if self.false_breakout_filter:
                # If the candle closes less than halfway past the breakout level, might be false
                breakout_strength = (latest_close - range_high) / range_high
                if breakout_strength < 0.005:  # Less than 0.5% past breakout
                    strength_factors.append(f"Weak breakout (+{breakout_strength*100:.2f}%)")
                    
                    if not volume_surge:
                        confirmed = False
                else:
                    strength_factors.append(f"Strong breakout (+{breakout_strength*100:.2f}%)")
            
            if confirmed:
                reasons = ["Bullish breakout above range high"]
                
                if volume_surge:
                    reasons.append(f"Volume surge ({latest_volume/avg_volume:.1f}x average)")
                    
                reasons.extend(strength_factors)
                
                # Calculate position size, stop loss and take profit
                stop_loss = min(latest_low, range_low)
                risk_amount = (latest_close - stop_loss) / latest_close
                
                # Adjust position size based on risk
                adjusted_position_size = self.position_size_pct
                if risk_amount > (self.stop_loss_pct / 100):
                    adjusted_position_size = self.position_size_pct * (self.stop_loss_pct / 100) / risk_amount
                
                return {
                    "symbol": symbol,
                    "side": "buy",
                    "price": latest_close,
                    "strength": 2 if volume_surge else 1,
                    "reasoning": ", ".join(reasons),
                    "stop_loss": stop_loss,
                    "take_profit": latest_close * (1 + self.profit_target_pct / 100),
                    "position_size_pct": adjusted_position_size,
                    "timestamp": datetime.now().isoformat(),
                    "id": f"breakout_buy_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
        # Check for bearish breakout (price breaks below range low)
        elif latest_close < range_low:
            # Confirm breakout with previous candles if required
            confirmed = True
            
            if self.confirmation_candles > 1:
                # Check if previous candles were inside the range
                for i in range(1, self.confirmation_candles):
                    prev_candle = confirm_candles[-(i+1)]
                    if prev_candle[4] < range_low:  # Previous close was also below
                        confirmed = False
                        break
            
            # Apply false breakout filter if enabled
            if self.false_breakout_filter:
                # If the candle closes less than halfway past the breakout level, might be false
                breakout_strength = (range_low - latest_close) / range_low
                if breakout_strength < 0.005:  # Less than 0.5% past breakout
                    strength_factors.append(f"Weak breakout (-{breakout_strength*100:.2f}%)")
                    
                    if not volume_surge:
                        confirmed = False
                else:
                    strength_factors.append(f"Strong breakout (-{breakout_strength*100:.2f}%)")
            
            if confirmed:
                reasons = ["Bearish breakout below range low"]
                
                if volume_surge:
                    reasons.append(f"Volume surge ({latest_volume/avg_volume:.1f}x average)")
                    
                reasons.extend(strength_factors)
                
                # Calculate position size, stop loss and take profit
                stop_loss = max(latest_high, range_high)
                risk_amount = (stop_loss - latest_close) / latest_close
                
                # Adjust position size based on risk
                adjusted_position_size = self.position_size_pct
                if risk_amount > (self.stop_loss_pct / 100):
                    adjusted_position_size = self.position_size_pct * (self.stop_loss_pct / 100) / risk_amount
                
                return {
                    "symbol": symbol,
                    "side": "sell",
                    "price": latest_close,
                    "strength": 2 if volume_surge else 1,
                    "reasoning": ", ".join(reasons),
                    "stop_loss": stop_loss,
                    "take_profit": latest_close * (1 - self.profit_target_pct / 100),
                    "position_size_pct": adjusted_position_size,
                    "timestamp": datetime.now().isoformat(),
                    "id": f"breakout_sell_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
            
            # Check stop loss and take profit
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
                    
            elif trade["side"] == "sell":
                # Check stop loss
                if latest_price >= trade["stop_loss"]:
                    trade["status"] = "closed"
                    trade["exit_price"] = latest_price
                    trade["exit_reason"] = "stop_loss"
                    trade["exit_time"] = datetime.now().isoformat()
                    
                    self._record_trade_performance(trade)
                    trades_to_remove.append(trade_id)
                    
                # Check take profit
                elif latest_price <= trade["take_profit"]:
                    trade["status"] = "closed"
                    trade["exit_price"] = latest_price
                    trade["exit_reason"] = "take_profit"
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
        self.logger.info("Training breakout strategy with historical data")
        
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
            for breakout_period in [10, 20, 30]:
                for volume_mult in [1.5, 2.0, 2.5]:
                    for confirmation in [1, 2, 3]:
                        for stop_loss in [5, 7, 10]:
                            params = {
                                "breakout_period": breakout_period,
                                "volume_multiplier_threshold": volume_mult,
                                "confirmation_candles": confirmation,
                                "stop_loss_pct": stop_loss
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
                    "breakout_period": self.breakout_period,
                    "volume_multiplier_threshold": self.volume_multiplier_threshold,
                    "confirmation_candles": self.confirmation_candles,
                    "stop_loss_pct": self.stop_loss_pct
                }
                
                self.breakout_period = best_params["breakout_period"]
                self.volume_multiplier_threshold = best_params["volume_multiplier_threshold"]
                self.confirmation_candles = best_params["confirmation_candles"]
                self.stop_loss_pct = best_params["stop_loss_pct"]
                
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
            self.logger.error(f"Error training breakout strategy: {str(e)}")
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
        # For now, just simulate some results based on the params
        
        # Simulated metric calculation based on parameters
        breakout_period = params["breakout_period"]
        volume_mult = params["volume_multiplier_threshold"]
        confirmation = params["confirmation_candles"]
        stop_loss = params["stop_loss_pct"]
        
        # Number of trades - More trades with shorter periods and less confirmation
        trade_frequency = (30 / breakout_period) * (3 / confirmation) * 10
        
        # Win rate factors
        # - More confirmation generally means higher accuracy but fewer trades
        # - Higher volume threshold generally means better confirmation
        # - Moderate breakout period (20) tends to be optimal
        base_win_rate = 40
        period_factor = 10 * (1 - abs(20 - breakout_period) / 20)  # 0-10 boost, optimal at 20
        confirmation_factor = confirmation * 3  # 3-9 boost
        volume_factor = (volume_mult - 1) * 10  # 5-15 boost
        
        expected_win_rate = base_win_rate + period_factor + confirmation_factor + volume_factor
        expected_win_rate = min(75, max(40, expected_win_rate))  # Cap between 40-75%
        
        # Profit and loss calculations
        avg_profit_pct = self.profit_target_pct * 0.7  # Assume we hit 70% of target on average
        avg_loss_pct = stop_loss
        
        # Expected profit per trade
        expected_profit_per_trade = (expected_win_rate / 100 * avg_profit_pct) - ((100 - expected_win_rate) / 100 * avg_loss_pct)
        
        # Simulate total profit
        total_profit = expected_profit_per_trade * trade_frequency
        
        # Return metrics
        return {
            "total_profit": total_profit,
            "win_rate": expected_win_rate,
            "trade_count": trade_frequency,
            "profit_per_trade": expected_profit_per_trade,
            "avg_profit": avg_profit_pct,
            "avg_loss": -avg_loss_pct,
            "initial_profit": 25  # baseline for comparison
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
            "ranges": self.ranges,
            "parameters": {
                "breakout_period": self.breakout_period,
                "volume_multiplier_threshold": self.volume_multiplier_threshold,
                "confirmation_candles": self.confirmation_candles,
                "false_breakout_filter": self.false_breakout_filter,
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
                
            if "ranges" in state:
                self.ranges = state["ranges"]
                
            if "parameters" in state:
                params = state["parameters"]
                for param, value in params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                        self.config[param] = value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading breakout strategy state: {str(e)}")
            return False 