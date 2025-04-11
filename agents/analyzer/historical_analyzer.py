"""
Historical Data Analyzer Agent - Analyzes historical market data and provides insights.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from services.market_data import MarketDataService

class HistoricalDataAnalyzerAgent(BaseAgent):
    """
    Historical Data Analyzer Agent that analyzes historical market data.
    
    This agent:
    - Collects historical data for all trading pairs
    - Performs technical and statistical analysis
    - Identifies patterns and trends
    - Provides insights to other agents
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the historical data analyzer.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="historical_analyzer")
        
        # Analysis configuration
        self.timeframes = config.get("timeframes", ["1d", "4h", "1h"])
        self.lookback_periods = config.get("lookback_periods", {
            "1d": 365,  # 1 year of daily data
            "4h": 90,   # 90 days of 4h data
            "1h": 30    # 30 days of 1h data
        })
        
        # Analysis parameters
        self.analysis_metrics = config.get("analysis_metrics", [
            "price_trend",
            "volume_trend",
            "volatility",
            "support_resistance",
            "correlation",
            "seasonality"
        ])
        
        # Data storage
        self.historical_data = {}
        self.analysis_results = {}
        self.last_analysis_time = None
        
        # Market data service
        self.market_data_service = MarketDataService(
            api_key=config.get("api_key"),
            secret=config.get("secret")
        )
        
        self.logger.info(f"Historical Data Analyzer initialized with {len(self.timeframes)} timeframes")
    
    def plan(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a plan for historical data analysis.
        
        Args:
            data: Input data for planning
            
        Returns:
            Dictionary containing the plan
        """
        # Get symbols to analyze
        symbols = data.get("symbols", []) if data else []
        if not symbols:
            symbols = self.market_data_service.get_available_symbols()
        
        plan = {
            "steps": [
                {
                    "action": "collect_data",
                    "symbols": symbols,
                    "timeframes": self.timeframes,
                    "lookback_periods": self.lookback_periods
                },
                {
                    "action": "analyze_data",
                    "metrics": self.analysis_metrics
                },
                {
                    "action": "update_insights",
                    "target_agents": data.get("target_agents", []) if data else []
                }
            ],
            "current_step": 0,
            "total_steps": 3,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        # Store plan in shared memory
        self.shared_memory.store_plan(self.id, plan)
        
        return plan
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step of the plan.
        
        Args:
            step: Step to execute
            
        Returns:
            Dictionary containing step execution results
        """
        action = step.get("action")
        
        if action == "collect_data":
            return self._collect_historical_data(step)
        elif action == "analyze_data":
            return self._analyze_historical_data(step)
        elif action == "update_insights":
            return self._update_insights(step)
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    def _collect_historical_data(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect historical data for the specified symbols and timeframes.
        
        Args:
            step: Step containing collection parameters
            
        Returns:
            Dictionary containing collection results
        """
        symbols = step.get("symbols", [])
        timeframes = step.get("timeframes", [])
        lookback_periods = step.get("lookback_periods", {})
        
        results = {}
        for symbol in symbols:
            results[symbol] = {}
            for timeframe in timeframes:
                try:
                    # Get lookback period
                    lookback = lookback_periods.get(timeframe, 30)
                    
                    # Fetch historical data
                    ohlcv = self.market_data_service.get_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=lookback
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    
                    # Store data
                    if symbol not in self.historical_data:
                        self.historical_data[symbol] = {}
                    self.historical_data[symbol][timeframe] = df
                    
                    results[symbol][timeframe] = {
                        "status": "success",
                        "data_points": len(df)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error collecting data for {symbol} {timeframe}: {str(e)}")
                    results[symbol][timeframe] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        return {"success": True, "results": results}
    
    def _analyze_historical_data(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze collected historical data.
        
        Args:
            step: Step containing analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        metrics = step.get("metrics", [])
        results = {}
        
        for symbol, timeframes in self.historical_data.items():
            results[symbol] = {}
            
            for timeframe, df in timeframes.items():
                results[symbol][timeframe] = {}
                
                # Calculate metrics
                for metric in metrics:
                    try:
                        if metric == "price_trend":
                            results[symbol][timeframe]["price_trend"] = self._calculate_price_trend(df)
                        elif metric == "volume_trend":
                            results[symbol][timeframe]["volume_trend"] = self._calculate_volume_trend(df)
                        elif metric == "volatility":
                            results[symbol][timeframe]["volatility"] = self._calculate_volatility(df)
                        elif metric == "support_resistance":
                            results[symbol][timeframe]["support_resistance"] = self._calculate_support_resistance(df)
                        elif metric == "correlation":
                            results[symbol][timeframe]["correlation"] = self._calculate_correlation(df)
                        elif metric == "seasonality":
                            results[symbol][timeframe]["seasonality"] = self._calculate_seasonality(df)
                            
                    except Exception as e:
                        self.logger.error(f"Error calculating {metric} for {symbol} {timeframe}: {str(e)}")
                        results[symbol][timeframe][metric] = {"error": str(e)}
        
        # Store results
        self.analysis_results = results
        self.last_analysis_time = datetime.now()
        
        return {"success": True, "results": results}
    
    def _update_insights(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update insights in shared memory for other agents.
        
        Args:
            step: Step containing target agents
            
        Returns:
            Dictionary containing update results
        """
        target_agents = step.get("target_agents", [])
        
        # Prepare insights
        insights = {
            "timestamp": datetime.now().isoformat(),
            "analysis_results": self.analysis_results,
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None
        }
        
        # Update shared memory
        self.shared_memory.update_system_data({
            "historical_insights": insights
        })
        
        return {"success": True, "updated_agents": target_agents}
    
    def _calculate_price_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price trend metrics."""
        # Calculate moving averages
        ma20 = df["close"].rolling(window=20).mean()
        ma50 = df["close"].rolling(window=50).mean()
        ma200 = df["close"].rolling(window=200).mean()
        
        # Calculate trend direction
        trend_direction = "up" if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1] else "down"
        
        # Calculate trend strength
        trend_strength = abs(ma20.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]
        
        return {
            "direction": trend_direction,
            "strength": trend_strength,
            "ma20": ma20.iloc[-1],
            "ma50": ma50.iloc[-1],
            "ma200": ma200.iloc[-1]
        }
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume trend metrics."""
        # Calculate volume moving average
        volume_ma = df["volume"].rolling(window=20).mean()
        
        # Calculate volume trend
        volume_trend = "increasing" if df["volume"].iloc[-1] > volume_ma.iloc[-1] else "decreasing"
        
        # Calculate volume strength
        volume_strength = df["volume"].iloc[-1] / volume_ma.iloc[-1]
        
        return {
            "trend": volume_trend,
            "strength": volume_strength,
            "current_volume": df["volume"].iloc[-1],
            "average_volume": volume_ma.iloc[-1]
        }
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility metrics."""
        # Calculate daily returns
        returns = df["close"].pct_change()
        
        # Calculate volatility (standard deviation of returns)
        volatility = returns.std()
        
        # Calculate ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        return {
            "volatility": volatility,
            "atr": atr.iloc[-1],
            "current_range": (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-1]
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels."""
        # Use pivot points
        pivot = (df["high"].iloc[-1] + df["low"].iloc[-1] + df["close"].iloc[-1]) / 3
        r1 = 2 * pivot - df["low"].iloc[-1]
        s1 = 2 * pivot - df["high"].iloc[-1]
        
        return {
            "pivot": pivot,
            "resistance": r1,
            "support": s1,
            "current_price": df["close"].iloc[-1]
        }
    
    def _calculate_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation with other assets."""
        # This would require data from other symbols
        # For now, return empty result
        return {"status": "not_implemented"}
    
    def _calculate_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate seasonal patterns."""
        # Calculate average returns by hour/day
        df["hour"] = df.index.hour
        df["day"] = df.index.dayofweek
        
        hourly_returns = df.groupby("hour")["close"].pct_change().mean()
        daily_returns = df.groupby("day")["close"].pct_change().mean()
        
        return {
            "hourly_pattern": hourly_returns.to_dict(),
            "daily_pattern": daily_returns.to_dict()
        }
    
    def get_insights(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """
        Get historical insights for a specific symbol and timeframe.
        
        Args:
            symbol: Optional symbol to filter by
            timeframe: Optional timeframe to filter by
            
        Returns:
            Dictionary containing insights
        """
        if not self.analysis_results:
            return {"error": "No analysis results available"}
        
        if symbol and timeframe:
            return self.analysis_results.get(symbol, {}).get(timeframe, {})
        elif symbol:
            return self.analysis_results.get(symbol, {})
        else:
            return self.analysis_results 