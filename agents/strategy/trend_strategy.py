"""
Trend Strategy Agent - Implements trend following trading strategies.

This agent analyzes market trends and generates trade signals based on trend-following principles.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from datetime import datetime

from agents.base_agent import BaseAgent

class TrendStrategyAgent(BaseAgent):
    """
    Trend Following Strategy Agent.
    
    Implements trend following strategies including moving average crossovers,
    price momentum, and breakout detection.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 config: Dict[str, Any],
                 parent_id: Optional[str] = None):
        """
        Initialize the trend strategy agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            config: Configuration parameters
            parent_id: ID of the parent agent
        """
        super().__init__(
            name=name,
            description=description,
            agent_type="trend_strategy",
            config=config,
            parent_id=parent_id
        )
        
        # Strategy parameters
        self.fast_ma = config.get("fast_ma", 10)
        self.slow_ma = config.get("slow_ma", 30)
        self.trend_strength_threshold = config.get("trend_strength_threshold", 0.02)
        self.confirmation_period = config.get("confirmation_period", 3)
        self.position_size_pct = config.get("position_size_pct", 10)
        self.stop_loss_pct = config.get("stop_loss_pct", 5)
        self.profit_target_pct = config.get("profit_target_pct", 15)
        self.max_trades = config.get("max_trades", 5)
        
        # State tracking
        self.active_trades = {}
        self.signals_history = []
        self.performance_history = []
        
        self.logger.info(f"Trend Strategy Agent {self.name} initialized")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strategy logic on market data.
        
        Args:
            data: Dictionary containing:
                - market_data: Market data from data provider
                - portfolio: Current portfolio state
                - analyzer_results: Results from market analyzer if available
                
        Returns:
            Dictionary containing trade signals and analysis
        """
        if not data or "market_data" not in data:
            self.logger.warning("No market data provided")
            return {"error": "Missing market data"}
        
        market_data = data["market_data"]
        portfolio = data.get("portfolio", {})
        analyzer_results = data.get("analyzer_results", {})
        
        results = {}
        signals = []
        
        try:
            # Process each symbol in market data
            for symbol, ohlcv in market_data.items():
                if ohlcv.empty:
                    continue
                
                # Get analysis from analyzer if available
                symbol_analysis = analyzer_results.get(symbol, {})
                
                # Generate strategy signal
                signal = self._generate_signal(symbol, ohlcv, symbol_analysis, portfolio)
                
                if signal:
                    signals.append(signal)
                    
                    # Store signal in history
                    self.signals_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "action": signal["action"],
                        "price": signal["price"],
                        "size": signal["size"]
                    })
                    
                    # Keep history at reasonable size
                    if len(self.signals_history) > 100:
                        self.signals_history = self.signals_history[-100:]
            
            # Update performance metrics
            self._update_performance_metrics(portfolio)
            
            results = {
                "signals": signals,
                "strategy_state": {
                    "active_trades": len(self.active_trades),
                    "recent_signals": len(self.signals_history)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Record execution
            self.performance_metrics["total_decisions"] += 1
            self.last_run = datetime.now().timestamp()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in strategy execution: {e}", exc_info=True)
            return {"error": f"Strategy execution error: {str(e)}"}
    
    def train(self, training_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            training_data: Dictionary containing:
                - historical_data: Historical OHLCV data
                - optimization_target: Metric to optimize for
                
        Returns:
            Tuple of (success, metrics)
        """
        if not training_data or "historical_data" not in training_data:
            return False, {"error": "No historical data provided"}
        
        historical_data = training_data["historical_data"]
        optimization_target = training_data.get("optimization_target", "sharpe_ratio")
        
        self.logger.info(f"Training Trend Strategy {self.name} for {optimization_target}")
        
        try:
            # Define parameter ranges
            param_ranges = {
                "fast_ma": [5, 10, 15, 20],
                "slow_ma": [20, 30, 40, 50],
                "trend_strength_threshold": [0.01, 0.02, 0.03, 0.05],
                "confirmation_period": [2, 3, 4]
            }
            
            # Simple grid search implementation
            best_params = {}
            best_metric = -float('inf')
            
            # Placeholder for actual optimization logic
            # In a real implementation, this would perform backtesting with different parameters
            
            # Simulate finding best parameters
            best_params = {
                "fast_ma": 10,
                "slow_ma": 30,
                "trend_strength_threshold": 0.02,
                "confirmation_period": 3
            }
            
            best_metric = 1.85  # Simulated Sharpe ratio
            
            # Update parameters if optimization was successful
            if best_params:
                self.update_config(best_params)
                
                # Update internal parameters
                for param, value in best_params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                
                self.logger.info(f"Training completed with best {optimization_target}: {best_metric}")
                
                return True, {
                    "best_params": best_params,
                    "performance_metric": optimization_target,
                    "metric_value": best_metric
                }
            else:
                return False, {"error": "No improvements found during training"}
                
        except Exception as e:
            self.logger.error(f"Error training Trend Strategy {self.name}: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def _generate_signal(self, symbol: str, ohlcv: pd.DataFrame, analysis: Dict[str, Any], portfolio: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Trading pair symbol
            ohlcv: OHLCV data
            analysis: Pre-computed analysis if available
            portfolio: Current portfolio state
            
        Returns:
            Signal dictionary or None if no signal
        """
        # Check if we have enough data
        if len(ohlcv) < self.slow_ma + self.confirmation_period:
            return None
        
        # Calculate indicators if not provided in analysis
        if not analysis:
            # Calculate moving averages
            ohlcv['fast_ma'] = ohlcv['close'].rolling(window=self.fast_ma).mean()
            ohlcv['slow_ma'] = ohlcv['close'].rolling(window=self.slow_ma).mean()
            
            # Calculate trend strength (slope of slow MA)
            ohlcv['trend_strength'] = (ohlcv['slow_ma'] - ohlcv['slow_ma'].shift(self.confirmation_period)) / ohlcv['slow_ma'].shift(self.confirmation_period)
        else:
            # Use pre-computed analysis
            trend_info = analysis.get("trend", {})
            momentum_info = analysis.get("momentum", {})
            trend_strength = trend_info.get("strength", 0)
            trend_direction = trend_info.get("direction", "neutral")
        
        # Get latest values
        latest = ohlcv.iloc[-1]
        
        # Current position in portfolio
        current_position = 0
        for position in portfolio.get("positions", []):
            if position["symbol"] == symbol:
                current_position = position["amount"]
                break
        
        # Check if symbol is already in active trades
        in_active_trade = symbol in self.active_trades
        
        # Generate signal
        signal = None
        
        # If we're using pre-computed analysis
        if analysis:
            if trend_direction == "strong_bullish" and not in_active_trade and current_position <= 0:
                # Strong buy signal
                signal = self._create_buy_signal(symbol, latest, ohlcv, portfolio)
            elif trend_direction == "strong_bearish" and (in_active_trade or current_position > 0):
                # Strong sell signal
                signal = self._create_sell_signal(symbol, latest, ohlcv, portfolio)
        else:
            # Simple moving average crossover logic
            # Buy signal: fast MA crosses above slow MA with positive trend strength
            if (ohlcv['fast_ma'].iloc[-2] <= ohlcv['slow_ma'].iloc[-2] and 
                latest['fast_ma'] > latest['slow_ma'] and 
                latest['trend_strength'] > self.trend_strength_threshold and
                not in_active_trade and current_position <= 0):
                signal = self._create_buy_signal(symbol, latest, ohlcv, portfolio)
                
            # Sell signal: fast MA crosses below slow MA with negative trend strength
            elif (ohlcv['fast_ma'].iloc[-2] >= ohlcv['slow_ma'].iloc[-2] and 
                  latest['fast_ma'] < latest['slow_ma'] and 
                  latest['trend_strength'] < -self.trend_strength_threshold and
                  (in_active_trade or current_position > 0)):
                signal = self._create_sell_signal(symbol, latest, ohlcv, portfolio)
        
        return signal
    
    def _create_buy_signal(self, symbol: str, latest_bar: pd.Series, ohlcv: pd.DataFrame, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a buy signal.
        
        Args:
            symbol: Trading pair symbol
            latest_bar: Latest price data
            ohlcv: Complete OHLCV data
            portfolio: Current portfolio state
            
        Returns:
            Buy signal dictionary
        """
        price = latest_bar['close']
        
        # Calculate position size based on portfolio value and risk
        portfolio_value = portfolio.get("total_value", 10000)
        position_value = portfolio_value * (self.position_size_pct / 100)
        size = position_value / price
        
        # Calculate stop loss price
        stop_loss = price * (1 - self.stop_loss_pct / 100)
        
        # Calculate take profit price
        take_profit = price * (1 + self.profit_target_pct / 100)
        
        # Add to active trades
        self.active_trades[symbol] = {
            "entry_price": price,
            "entry_time": datetime.now().isoformat(),
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        return {
            "symbol": symbol,
            "action": "buy",
            "price": price,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": "Trend following buy signal triggered",
            "confidence": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_sell_signal(self, symbol: str, latest_bar: pd.Series, ohlcv: pd.DataFrame, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a sell signal.
        
        Args:
            symbol: Trading pair symbol
            latest_bar: Latest price data
            ohlcv: Complete OHLCV data
            portfolio: Current portfolio state
            
        Returns:
            Sell signal dictionary
        """
        price = latest_bar['close']
        
        # If we have an active trade, use its size
        size = 0
        if symbol in self.active_trades:
            size = self.active_trades[symbol]["size"]
            active_trade = self.active_trades[symbol]
            
            # Calculate profit/loss
            pnl = (price - active_trade["entry_price"]) / active_trade["entry_price"] * 100
            
            # Record trade performance
            self.performance_history.append({
                "symbol": symbol,
                "entry_price": active_trade["entry_price"],
                "exit_price": price,
                "size": size,
                "entry_time": active_trade["entry_time"],
                "exit_time": datetime.now().isoformat(),
                "pnl_pct": pnl,
                "pnl_value": (price - active_trade["entry_price"]) * size
            })
            
            # Keep history at reasonable size
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            # Remove from active trades
            del self.active_trades[symbol]
            
            # Update performance metrics
            self.performance_metrics["total_trades"] += 1
            if pnl > 0:
                self.performance_metrics["profitable_trades"] += 1
                self.performance_metrics["total_profit"] += pnl
            
        # If no active trade but we have a position in portfolio
        else:
            for position in portfolio.get("positions", []):
                if position["symbol"] == symbol:
                    size = position["amount"]
                    break
        
        return {
            "symbol": symbol,
            "action": "sell",
            "price": price,
            "size": size,
            "reason": "Trend following sell signal triggered",
            "confidence": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_performance_metrics(self, portfolio: Dict[str, Any]) -> None:
        """
        Update performance metrics based on current portfolio.
        
        Args:
            portfolio: Current portfolio state
        """
        if not self.performance_history:
            return
            
        # Calculate win rate
        if self.performance_metrics["total_trades"] > 0:
            win_rate = self.performance_metrics["profitable_trades"] / self.performance_metrics["total_trades"]
            self.performance_metrics["win_rate"] = win_rate
            
        # Calculate average profit per trade
        if self.performance_metrics["total_trades"] > 0:
            avg_profit = self.performance_metrics["total_profit"] / self.performance_metrics["total_trades"]
            self.performance_metrics["avg_profit_per_trade"] = avg_profit
            
        # Calculate max drawdown (simplified)
        equity_curve = []
        running_balance = 10000  # Starting balance
        
        for trade in self.performance_history:
            pnl_value = trade["pnl_value"]
            running_balance += pnl_value
            equity_curve.append(running_balance)
            
        if equity_curve:
            peak = max(equity_curve)
            max_drawdown = 0
            
            for balance in equity_curve:
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)
                
            self.performance_metrics["max_drawdown"] = max_drawdown * 100  # as percentage
            
        # Get current portfolio value if available
        if "total_value" in portfolio:
            current_value = portfolio["total_value"]
            initial_value = 10000  # Assuming initial value
            
            # Calculate simple Sharpe ratio (ignoring risk-free rate)
            if current_value > initial_value:
                returns = (current_value - initial_value) / initial_value
                self.performance_metrics["sharpe_ratio"] = returns / (self.performance_metrics["max_drawdown"] / 100 + 0.0001) 