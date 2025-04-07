 n  """
Risk Management Agent - Monitors and manages trading system risks.

This agent is responsible for monitoring risk metrics, enforcing risk limits,
and providing risk assessments to protect the trading portfolio.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from agents.base_agent import BaseAgent

class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent responsible for monitoring and managing trading risks.
    
    This agent enforces risk controls, calculates risk metrics, and can 
    implement emergency measures to protect the portfolio during adverse
    market conditions.
    """
    
    def __init__(self, agent_id: str, name: str, description: str, config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the risk manager agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Description of the agent's purpose
            config: Configuration parameters
            parent_id: ID of the parent agent if any
        """
        super().__init__(agent_id, name, description, "risk_manager", config, parent_id)
        
        # Initialize risk parameters from config
        self.max_portfolio_drawdown = self.config.get("max_portfolio_drawdown", 25)  # percent
        self.max_single_asset_allocation = self.config.get("max_single_asset_allocation", 40)  # percent
        self.intraday_loss_limit = self.config.get("intraday_loss_limit", 5)  # percent
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)
        self.volatility_scaling = self.config.get("volatility_scaling", True)
        self.risk_free_rate = self.config.get("risk_free_rate", 2.0) / 100.0  # convert to decimal
        self.risk_metrics_to_track = self.config.get("risk_metrics", ["sharpe", "sortino", "max_drawdown"])
        self.var_confidence_level = self.config.get("var_confidence_level", 95)
        self.stress_test_scenarios = self.config.get("stress_test_scenarios", ["2018_bear", "2020_crash"])
        
        # Risk state
        self.current_risk_level = "normal"  # normal, elevated, high, critical
        self.active_warnings = []
        self.risk_metrics = {}
        self.drawdown_history = []
        self.portfolio_var = 0.0  # Value at Risk
        
        # Historical data
        self.daily_returns = []
        self.portfolio_values = []
        self.asset_allocations = {}
        
        # Last risk assessment timestamp
        self.last_assessment = None
        
        self.logger = logging.getLogger(f"risk_manager_{agent_id}")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform risk assessment and management based on current portfolio state.
        
        Args:
            data: Dictionary containing portfolio and market data
                 Expected format:
                 {
                     "portfolio": {
                         "total_value": 10000,
                         "starting_value": 9500,
                         "holdings": {"BTC": 0.1, "ETH": 1.5},
                         "asset_values": {"BTC": 5000, "ETH": 4500},
                         "cash": 500
                     },
                     "market_data": {
                         "BTC/USDT": {...},  # Market data by symbol
                         "ETH/USDT": {...},
                         ...
                     },
                     "trades": [...],  # Recent trades
                     "positions": [...]  # Current positions
                 }
        
        Returns:
            Dictionary with risk assessment and actions:
            {
                "risk_level": "normal"|"elevated"|"high"|"critical",
                "warnings": [...],
                "actions": [...],
                "metrics": {...},
                "limits": {...}
            }
        """
        if not self.is_active:
            return {"error": "Agent is not active"}
        
        try:
            self.logger.info("Running risk manager assessment")
            self.last_assessment = datetime.now().isoformat()
            
            # Extract data
            portfolio = data.get("portfolio", {})
            market_data = data.get("market_data", {})
            trades = data.get("trades", [])
            positions = data.get("positions", [])
            
            # Update portfolio history
            self._update_portfolio_history(portfolio)
            
            # Calculate risk metrics
            self._calculate_risk_metrics(portfolio, market_data)
            
            # Check for risk violations
            warnings = self._check_risk_limits(portfolio)
            
            # Determine overall risk level
            risk_level = self._determine_risk_level(warnings, portfolio)
            
            # Generate risk actions
            actions = self._generate_risk_actions(risk_level, warnings, portfolio)
            
            # Calculate position limits
            position_limits = self._calculate_position_limits(portfolio, market_data)
            
            # Prepare response
            response = {
                "risk_level": risk_level,
                "warnings": warnings,
                "actions": actions,
                "metrics": self.risk_metrics,
                "limits": position_limits,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update current state
            self.current_risk_level = risk_level
            self.active_warnings = warnings
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in risk manager run: {str(e)}")
            return {
                "error": str(e),
                "risk_level": "unknown",
                "warnings": ["Risk assessment failed"],
                "actions": ["Proceed with caution, risk assessment unavailable"],
                "metrics": {},
                "limits": {}
            }
    
    def _update_portfolio_history(self, portfolio: Dict[str, Any]) -> None:
        """Update portfolio history with current portfolio state."""
        current_value = portfolio.get("total_value", 0)
        if current_value > 0:
            self.portfolio_values.append(current_value)
            
            # Calculate daily return if we have at least 2 values
            if len(self.portfolio_values) >= 2:
                daily_return = (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
                self.daily_returns.append(daily_return)
                
            # Update asset allocations
            total_value = portfolio.get("total_value", 0)
            if total_value > 0:
                asset_values = portfolio.get("asset_values", {})
                self.asset_allocations = {
                    asset: value / total_value * 100 for asset, value in asset_values.items()
                }
    
    def _calculate_risk_metrics(self, portfolio: Dict[str, Any], market_data: Dict[str, Any]) -> None:
        """Calculate various risk metrics based on portfolio and market data."""
        # Only calculate if we have enough data
        if len(self.daily_returns) < 5:
            self.risk_metrics = {"insufficient_data": True}
            return
            
        returns = np.array(self.daily_returns)
        
        # Maximum Drawdown
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        # Update drawdown history
        current_dd = drawdown[-1]
        self.drawdown_history.append(current_dd)
        
        # Sharpe Ratio (annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        annualized_sharpe = (mean_return - self.risk_free_rate/252) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino Ratio (annualized)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0001
        annualized_sortino = (mean_return - self.risk_free_rate/252) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Value at Risk (VaR)
        var_percentile = 100 - self.var_confidence_level
        var = np.percentile(returns, var_percentile)
        portfolio_value = portfolio.get("total_value", 0)
        var_amount = portfolio_value * abs(var)
        
        # Volatility (annualized)
        annualized_volatility = std_return * np.sqrt(252) * 100  # as percentage
        
        # Portfolio concentration
        asset_values = portfolio.get("asset_values", {})
        total_value = portfolio.get("total_value", 0)
        if total_value > 0:
            largest_allocation = max(value / total_value * 100 for value in asset_values.values()) if asset_values else 0
        else:
            largest_allocation = 0
            
        # Store calculated metrics
        self.risk_metrics = {
            "max_drawdown": max_drawdown,
            "current_drawdown": current_dd,
            "sharpe_ratio": annualized_sharpe,
            "sortino_ratio": annualized_sortino,
            "value_at_risk": var_amount,
            "value_at_risk_pct": abs(var) * 100,
            "annualized_volatility": annualized_volatility,
            "largest_allocation": largest_allocation,
            "allocation_count": len(asset_values),
            "profit_factor": self._calculate_profit_factor(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store VaR for use elsewhere
        self.portfolio_var = var_amount
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (sum of profits / sum of losses)."""
        if not self.daily_returns:
            return 0
            
        returns = np.array(self.daily_returns)
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        sum_profits = np.sum(profits) if len(profits) > 0 else 0
        sum_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0001
        
        return sum_profits / sum_losses
    
    def _check_risk_limits(self, portfolio: Dict[str, Any]) -> List[str]:
        """Check for risk limit violations and return warnings."""
        warnings = []
        
        # Check maximum drawdown
        if (self.risk_metrics.get("current_drawdown", 0) < 0 and
                abs(self.risk_metrics.get("current_drawdown", 0)) > self.max_portfolio_drawdown):
            warnings.append(f"Portfolio drawdown ({abs(self.risk_metrics.get('current_drawdown', 0)):.2f}%) exceeds maximum ({self.max_portfolio_drawdown}%)")
        
        # Check asset concentration
        if self.risk_metrics.get("largest_allocation", 0) > self.max_single_asset_allocation:
            warnings.append(f"Asset concentration ({self.risk_metrics.get('largest_allocation', 0):.2f}%) exceeds maximum ({self.max_single_asset_allocation}%)")
        
        # Check daily loss limit
        if self.daily_returns and self.daily_returns[-1] < 0 and abs(self.daily_returns[-1]) * 100 > self.intraday_loss_limit:
            warnings.append(f"Daily loss ({abs(self.daily_returns[-1]) * 100:.2f}%) exceeds limit ({self.intraday_loss_limit}%)")
        
        # Check Value at Risk
        if portfolio.get("total_value", 0) > 0:
            var_pct = self.risk_metrics.get("value_at_risk_pct", 0)
            if var_pct > self.intraday_loss_limit * 1.5:  # VaR should be less than 1.5x daily loss limit
                warnings.append(f"Value at Risk ({var_pct:.2f}%) exceeds acceptable level ({self.intraday_loss_limit * 1.5:.2f}%)")
        
        # Check volatility
        if self.risk_metrics.get("annualized_volatility", 0) > 50:  # 50% annualized volatility is high
            warnings.append(f"Portfolio volatility ({self.risk_metrics.get('annualized_volatility', 0):.2f}%) is elevated")
        
        # Portfolio health metrics
        if self.risk_metrics.get("sharpe_ratio", 0) < 0.5:
            warnings.append(f"Low Sharpe ratio ({self.risk_metrics.get('sharpe_ratio', 0):.2f})")
        
        if self.risk_metrics.get("profit_factor", 0) < 1.0:
            warnings.append(f"Profit factor below 1.0 ({self.risk_metrics.get('profit_factor', 0):.2f})")
            
        return warnings
    
    def _determine_risk_level(self, warnings: List[str], portfolio: Dict[str, Any]) -> str:
        """Determine overall risk level based on warnings and metrics."""
        if not warnings:
            return "normal"
            
        # Count severe warnings
        severe_count = 0
        for warning in warnings:
            if "exceeds maximum" in warning or "exceeds limit" in warning:
                severe_count += 1
        
        # Check drawdown specifically
        current_drawdown = abs(self.risk_metrics.get("current_drawdown", 0))
        
        if severe_count >= 3 or current_drawdown > self.max_portfolio_drawdown * 1.5:
            return "critical"
        elif severe_count >= 2 or current_drawdown > self.max_portfolio_drawdown:
            return "high"
        elif severe_count >= 1 or current_drawdown > self.max_portfolio_drawdown * 0.7:
            return "elevated"
        else:
            return "normal"
    
    def _generate_risk_actions(self, risk_level: str, warnings: List[str], portfolio: Dict[str, Any]) -> List[str]:
        """Generate recommended actions based on risk level and warnings."""
        actions = []
        
        if risk_level == "normal":
            actions.append("Continue normal trading operations")
            
        elif risk_level == "elevated":
            actions.append("Reduce position sizes by 25%")
            actions.append("Increase cash reserves")
            
            # Add specific actions based on warnings
            for warning in warnings:
                if "asset concentration" in warning:
                    actions.append("Reduce exposure to largest holdings")
                if "volatility" in warning:
                    actions.append("Favor less volatile assets")
            
        elif risk_level == "high":
            actions.append("Reduce position sizes by 50%")
            actions.append("Tighten stop-loss levels")
            actions.append("Temporarily pause opening new positions")
            
            # Check for specific high-risk conditions
            if self.risk_metrics.get("current_drawdown", 0) < -15:
                actions.append("Consider hedging with inverse positions")
                
            for warning in warnings:
                if "daily loss" in warning:
                    actions.append("Close underperforming positions")
            
        elif risk_level == "critical":
            actions.append("Reduce position sizes by 75% or more")
            actions.append("Close all losing positions")
            actions.append("Increase cash position to at least 50%")
            actions.append("Suspend all new trades until risk level decreases")
            
            if self.risk_metrics.get("current_drawdown", 0) < -20:
                actions.append("Consider suspending all trading activity")
        
        return actions
    
    def _calculate_position_limits(self, portfolio: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position limits for each asset based on risk parameters."""
        limits = {}
        
        # Extract portfolio value and apply risk scaling factor
        portfolio_value = portfolio.get("total_value", 0)
        risk_factor = self._get_risk_scaling_factor()
        
        # Default maximum position size (adjusted by risk level)
        default_max_position_pct = self.max_single_asset_allocation * risk_factor
        
        # Calculate limits for each asset
        asset_values = portfolio.get("asset_values", {})
        for asset, value in asset_values.items():
            # Current allocation percentage
            current_pct = (value / portfolio_value * 100) if portfolio_value > 0 else 0
            
            # Determine max allocation percentage based on risk level
            max_allocation_pct = min(default_max_position_pct, self.max_single_asset_allocation)
            
            # Calculate how much more can be allocated to this asset
            remaining_allocation_pct = max(0, max_allocation_pct - current_pct)
            remaining_allocation_value = portfolio_value * (remaining_allocation_pct / 100)
            
            # Calculate recommended allocation based on risk level
            if self.current_risk_level == "normal":
                target_allocation_pct = current_pct
            elif self.current_risk_level == "elevated":
                target_allocation_pct = current_pct * 0.75
            elif self.current_risk_level == "high":
                target_allocation_pct = current_pct * 0.5
            else:  # critical
                target_allocation_pct = current_pct * 0.25
            
            # Store limits
            limits[asset] = {
                "current_allocation_pct": current_pct,
                "max_allocation_pct": max_allocation_pct,
                "remaining_allocation_pct": remaining_allocation_pct,
                "remaining_allocation_value": remaining_allocation_value,
                "target_allocation_pct": target_allocation_pct,
                "target_allocation_value": portfolio_value * (target_allocation_pct / 100)
            }
        
        return limits
    
    def _get_risk_scaling_factor(self) -> float:
        """
        Get a scaling factor for risk based on current risk level.
        
        Returns a value between 0.0 and 1.0, where lower values 
        indicate more risk reduction.
        """
        if self.current_risk_level == "normal":
            return 1.0
        elif self.current_risk_level == "elevated":
            return 0.75
        elif self.current_risk_level == "high":
            return 0.5
        else:  # critical
            return 0.25
    
    def run_stress_test(self, portfolio: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """
        Run a stress test on the portfolio using historical scenarios.
        
        Args:
            portfolio: Current portfolio state
            scenario: Name of the stress test scenario
            
        Returns:
            Dictionary with stress test results
        """
        self.logger.info(f"Running stress test: {scenario}")
        
        try:
            # This is a simplified stress test implementation
            scenario_data = self._get_stress_test_data(scenario)
            
            if not scenario_data:
                return {
                    "success": False,
                    "error": f"Scenario data not available for {scenario}"
                }
            
            # Extract portfolio assets
            asset_values = portfolio.get("asset_values", {})
            portfolio_value = portfolio.get("total_value", 0)
            
            # Apply scenario price changes to portfolio
            new_values = {}
            for asset, value in asset_values.items():
                change_pct = scenario_data.get(asset, scenario_data.get("default", -10)) / 100
                new_values[asset] = value * (1 + change_pct)
            
            # Calculate portfolio impact
            new_portfolio_value = sum(new_values.values()) + portfolio.get("cash", 0)
            impact_pct = ((new_portfolio_value / portfolio_value) - 1) * 100 if portfolio_value > 0 else 0
            
            # Determine if the portfolio would survive this scenario
            max_drawdown = abs(self.max_portfolio_drawdown)
            would_hit_max_drawdown = abs(impact_pct) > max_drawdown
            
            # Calculate VaR adjustment
            var_adjustment = 1 + (abs(impact_pct) / 100)
            
            return {
                "success": True,
                "scenario": scenario,
                "current_portfolio_value": portfolio_value,
                "stressed_portfolio_value": new_portfolio_value,
                "impact_pct": impact_pct,
                "would_hit_max_drawdown": would_hit_max_drawdown,
                "var_adjustment_factor": var_adjustment,
                "asset_impacts": {
                    asset: scenario_data.get(asset, scenario_data.get("default", -10))
                    for asset in asset_values
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in stress test: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_stress_test_data(self, scenario: str) -> Dict[str, float]:
        """
        Get price change data for the specified stress test scenario.
        
        Returns dictionary mapping assets to percentage price changes.
        """
        # In a real implementation, this would load from historical data
        # These are simplified example values
        scenarios = {
            "2018_bear": {
                "BTC/USDT": -65,
                "ETH/USDT": -80,
                "default": -70
            },
            "2020_crash": {
                "BTC/USDT": -50,
                "ETH/USDT": -60,
                "default": -55
            },
            "2021_may_crash": {
                "BTC/USDT": -40,
                "ETH/USDT": -45,
                "default": -42
            }
        }
        
        return scenarios.get(scenario, {})
    
    def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the risk manager using historical data.
        
        Args:
            training_data: Historical portfolio and market data
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training risk manager")
        
        try:
            # Extract historical returns
            historical_returns = training_data.get("historical_returns", [])
            historical_drawdowns = training_data.get("historical_drawdowns", [])
            
            if not historical_returns or len(historical_returns) < 30:
                return {
                    "success": False,
                    "error": "Insufficient historical data for training"
                }
            
            # Calculate optimal risk parameters based on historical data
            returns = np.array(historical_returns)
            drawdowns = np.array(historical_drawdowns) if historical_drawdowns else None
            
            # Calculate VaR at different confidence levels
            var_90 = abs(np.percentile(returns, 10))
            var_95 = abs(np.percentile(returns, 5))
            var_99 = abs(np.percentile(returns, 1))
            
            # Determine optimal maximum drawdown
            if drawdowns is not None and len(drawdowns) > 0:
                max_historical_dd = abs(np.min(drawdowns))
                optimal_max_dd = max(self.max_portfolio_drawdown, max_historical_dd * 1.2)
            else:
                optimal_max_dd = self.max_portfolio_drawdown
            
            # Determine optimal intraday loss limit
            daily_loss_95 = var_95 * 100  # Convert to percentage
            optimal_loss_limit = max(self.intraday_loss_limit, daily_loss_95 * 1.2)
            
            # Update parameters
            old_params = {
                "max_portfolio_drawdown": self.max_portfolio_drawdown,
                "intraday_loss_limit": self.intraday_loss_limit,
                "var_confidence_level": self.var_confidence_level
            }
            
            # Only update if significant improvement
            if optimal_max_dd > self.max_portfolio_drawdown * 1.1 or optimal_max_dd < self.max_portfolio_drawdown * 0.9:
                self.max_portfolio_drawdown = optimal_max_dd
                self.config["max_portfolio_drawdown"] = optimal_max_dd
            
            if optimal_loss_limit > self.intraday_loss_limit * 1.1 or optimal_loss_limit < self.intraday_loss_limit * 0.9:
                self.intraday_loss_limit = optimal_loss_limit
                self.config["intraday_loss_limit"] = optimal_loss_limit
            
            # Determine best VaR confidence level
            var_levels = {90: var_90, 95: var_95, 99: var_99}
            best_var_level = min(var_levels.items(), key=lambda x: abs(x[1] * 100 - optimal_loss_limit))[0]
            
            if best_var_level != self.var_confidence_level:
                self.var_confidence_level = best_var_level
                self.config["var_confidence_level"] = best_var_level
            
            new_params = {
                "max_portfolio_drawdown": self.max_portfolio_drawdown,
                "intraday_loss_limit": self.intraday_loss_limit,
                "var_confidence_level": self.var_confidence_level
            }
            
            return {
                "success": True,
                "old_params": old_params,
                "new_params": new_params,
                "var_metrics": {
                    "90": var_90 * 100,
                    "95": var_95 * 100,
                    "99": var_99 * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training risk manager: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return self.risk_metrics
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the risk manager.
        
        Returns:
            Dictionary with state data to be persisted
        """
        state = {
            "risk_metrics": self.risk_metrics,
            "current_risk_level": self.current_risk_level,
            "active_warnings": self.active_warnings,
            "portfolio_values": self.portfolio_values[-100:] if self.portfolio_values else [],
            "daily_returns": self.daily_returns[-100:] if self.daily_returns else [],
            "drawdown_history": self.drawdown_history[-100:] if self.drawdown_history else [],
            "asset_allocations": self.asset_allocations,
            "last_assessment": self.last_assessment,
            "config": {
                "max_portfolio_drawdown": self.max_portfolio_drawdown,
                "max_single_asset_allocation": self.max_single_asset_allocation,
                "intraday_loss_limit": self.intraday_loss_limit,
                "correlation_threshold": self.correlation_threshold,
                "var_confidence_level": self.var_confidence_level
            }
        }
        return state
    
    def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load a previously saved state.
        
        Args:
            state: Previously saved state data
            
        Returns:
            Boolean indicating success
        """
        try:
            if "risk_metrics" in state:
                self.risk_metrics = state["risk_metrics"]
                
            if "current_risk_level" in state:
                self.current_risk_level = state["current_risk_level"]
                
            if "active_warnings" in state:
                self.active_warnings = state["active_warnings"]
                
            if "portfolio_values" in state:
                self.portfolio_values = state["portfolio_values"]
                
            if "daily_returns" in state:
                self.daily_returns = state["daily_returns"]
                
            if "drawdown_history" in state:
                self.drawdown_history = state["drawdown_history"]
                
            if "asset_allocations" in state:
                self.asset_allocations = state["asset_allocations"]
                
            if "last_assessment" in state:
                self.last_assessment = state["last_assessment"]
                
            if "config" in state:
                config = state["config"]
                for key, value in config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        self.config[key] = value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading risk manager state: {str(e)}")
            return False 