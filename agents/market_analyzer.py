"""
Market Analyzer Agent - Specialized in analyzing market data.

This agent is responsible for analyzing market data and providing insights.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent

class MarketAnalyzerAgent(BaseAgent):
    """
    Agent specialized in market data analysis.
    
    Analyzes market data to identify patterns, trends, and potential
    trading opportunities using technical analysis.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 config: Dict[str, Any],
                 parent_id: Optional[str] = None):
        """
        Initialize the market analyzer agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            config: Configuration parameters
            parent_id: ID of the parent agent
        """
        super().__init__(
            name=name,
            description=description,
            agent_type="market_analyzer",
            config=config,
            parent_id=parent_id
        )
        
        # Initialize technical analysis parameters
        self.short_ma_period = config.get("short_ma_period", 20)
        self.medium_ma_period = config.get("medium_ma_period", 50)
        self.long_ma_period = config.get("long_ma_period", 200)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        self.bb_period = config.get("bb_period", 20)
        self.bb_std_dev = config.get("bb_std_dev", 2)
        
        # Initialize analyzers
        self.analyzers = {
            "trend": self._analyze_trend,
            "momentum": self._analyze_momentum,
            "volatility": self._analyze_volatility,
            "support_resistance": self._analyze_support_resistance,
        }
        
        self.logger.info(f"Market Analyzer Agent {self.name} initialized")
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and provide insights.
        
        Args:
            data: Dictionary containing market data with keys:
                - symbol: Trading pair symbol
                - ohlcv: OHLCV data as DataFrame
                - timeframe: Timeframe of the data
                
        Returns:
            Dictionary containing analysis results
        """
        if not data or "ohlcv" not in data or data["ohlcv"].empty:
            self.logger.warning("No data provided for analysis")
            return {"error": "No data available for analysis"}
        
        symbol = data.get("symbol", "UNKNOWN")
        timeframe = data.get("timeframe", "1h")
        df = data["ohlcv"].copy()
        
        self.logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
        
        # Ensure we have enough data
        if len(df) < max(self.long_ma_period, self.bb_period + 50):
            self.logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            return {"error": "Insufficient data for complete analysis"}
        
        # Calculate indicators
        try:
            df = self._calculate_indicators(df)
            
            # Run all analyzers
            results = {}
            for analyzer_name, analyzer_func in self.analyzers.items():
                results[analyzer_name] = analyzer_func(df, symbol, timeframe)
            
            # Add summary
            results["summary"] = self._generate_summary(results, df, symbol, timeframe)
            
            # Record the analysis in performance metrics
            self.performance_metrics["total_decisions"] += 1
            
            # Update the last run time
            self.last_run = datetime.now().timestamp()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
            return {"error": f"Analysis error: {str(e)}"}
    
    def train(self, training_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Train the agent by optimizing technical parameters.
        
        Args:
            training_data: Dictionary containing:
                - historical_data: Dictionary mapping symbols to OHLCV DataFrames
                - performance_metric: Metric to optimize for (e.g., "sharpe_ratio")
                - optimization_period: Days of data to use for optimization
                
        Returns:
            Tuple of (success, metrics)
        """
        if not training_data or "historical_data" not in training_data:
            return False, {"error": "No historical data provided"}
        
        historical_data = training_data["historical_data"]
        performance_metric = training_data.get("performance_metric", "sharpe_ratio")
        optimization_period = training_data.get("optimization_period", 90)
        
        self.logger.info(f"Training Market Analyzer {self.name} for {performance_metric}")
        
        try:
            # Define parameter ranges
            param_ranges = {
                "short_ma_period": range(5, 30, 5),
                "medium_ma_period": range(30, 80, 10),
                "rsi_period": range(7, 21, 7),
                "macd_fast": range(8, 16, 2),
                "macd_slow": range(20, 32, 3),
            }
            
            best_params = {}
            best_metric = -float('inf')
            
            # Perform grid search
            # In a real system, use more sophisticated methods like Bayesian optimization
            for symbol, ohlcv_data in historical_data.items():
                if len(ohlcv_data) < 500:  # Need enough data
                    continue
                    
                # Placeholder for actual optimization logic
                # In a real system, this would be a comprehensive backtest
                # For brevity, we'll simulate the optimization
                
                # Simulate finding best parameters
                best_params = {
                    "short_ma_period": 15,
                    "medium_ma_period": 50,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                }
                
                best_metric = 1.75  # Simulated Sharpe ratio
            
            # Update parameters if optimization was successful
            if best_params:
                self.update_config(best_params)
                
                # Update internal parameters
                for param, value in best_params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                
                self.logger.info(f"Training completed with best {performance_metric}: {best_metric}")
                
                return True, {
                    "best_params": best_params,
                    "performance_metric": performance_metric,
                    "metric_value": best_metric
                }
            else:
                return False, {"error": "No improvements found during training"}
                
        except Exception as e:
            self.logger.error(f"Error training Market Analyzer {self.name}: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with indicators added
        """
        # Moving Averages
        df['sma_short'] = df['close'].rolling(window=self.short_ma_period).mean()
        df['sma_medium'] = df['close'].rolling(window=self.medium_ma_period).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_ma_period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema_fast'] = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bb_std_dev)
        
        # Additional indicators
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            np.maximum(
                df['high'] - df['low'],
                np.abs(df['high'] - df['close'].shift())
            ),
            np.abs(df['low'] - df['close'].shift())
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def _analyze_trend(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market trend.
        
        Args:
            df: DataFrame with indicators
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary with trend analysis
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Determine trend direction
        trend_direction = "neutral"
        trend_strength = 0.0
        
        if latest['sma_short'] > latest['sma_medium'] > latest['sma_long']:
            trend_direction = "strong_bullish"
            trend_strength = 1.0
        elif latest['sma_short'] > latest['sma_medium']:
            trend_direction = "bullish"
            trend_strength = 0.5
        elif latest['sma_short'] < latest['sma_medium'] < latest['sma_long']:
            trend_direction = "strong_bearish"
            trend_strength = -1.0
        elif latest['sma_short'] < latest['sma_medium']:
            trend_direction = "bearish"
            trend_strength = -0.5
        
        # Check for crossovers
        short_medium_crossover = (
            (prev['sma_short'] <= prev['sma_medium'] and latest['sma_short'] > latest['sma_medium']) or
            (prev['sma_short'] >= prev['sma_medium'] and latest['sma_short'] < latest['sma_medium'])
        )
        
        medium_long_crossover = (
            (prev['sma_medium'] <= prev['sma_long'] and latest['sma_medium'] > latest['sma_long']) or
            (prev['sma_medium'] >= prev['sma_long'] and latest['sma_medium'] < latest['sma_long'])
        )
        
        return {
            "direction": trend_direction,
            "strength": trend_strength,
            "short_medium_crossover": short_medium_crossover,
            "medium_long_crossover": medium_long_crossover,
            "price_above_long_ma": latest['close'] > latest['sma_long'],
            "price_relative_to_trend": latest['close'] / latest['sma_medium'] - 1
        }
    
    def _analyze_momentum(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market momentum.
        
        Args:
            df: DataFrame with indicators
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary with momentum analysis
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # RSI conditions
        rsi_condition = "neutral"
        if latest['rsi'] > self.rsi_overbought:
            rsi_condition = "overbought"
        elif latest['rsi'] < self.rsi_oversold:
            rsi_condition = "oversold"
        
        # MACD conditions
        macd_signal_crossover = (
            (prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']) or
            (prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal'])
        )
        
        macd_direction = "neutral"
        if latest['macd'] > 0 and latest['macd_histogram'] > 0:
            macd_direction = "bullish"
        elif latest['macd'] < 0 and latest['macd_histogram'] < 0:
            macd_direction = "bearish"
        
        return {
            "rsi_value": latest['rsi'],
            "rsi_condition": rsi_condition,
            "macd_value": latest['macd'],
            "macd_signal": latest['macd_signal'],
            "macd_histogram": latest['macd_histogram'],
            "macd_direction": macd_direction,
            "macd_signal_crossover": macd_signal_crossover,
            "momentum_strength": (latest['rsi'] - 50) / 50  # Normalized momentum
        }
    
    def _analyze_volatility(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market volatility.
        
        Args:
            df: DataFrame with indicators
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary with volatility analysis
        """
        latest = df.iloc[-1]
        
        # Bollinger Bands width (volatility indicator)
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
        
        # Relative position within Bollinger Bands
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        
        # ATR relative to price
        atr_percent = latest['atr'] / latest['close'] * 100
        
        # Historical volatility (20-day standard deviation of returns)
        returns = df['close'].pct_change().dropna()
        hist_volatility = returns.rolling(window=20).std().iloc[-1] * 100
        
        # Interpret volatility
        volatility_condition = "normal"
        if bb_width > 0.08:  # Threshold can be adjusted
            volatility_condition = "high"
        elif bb_width < 0.02:  # Threshold can be adjusted
            volatility_condition = "low"
        
        return {
            "bb_width": bb_width,
            "bb_position": bb_position,
            "atr": latest['atr'],
            "atr_percent": atr_percent,
            "historical_volatility": hist_volatility,
            "volatility_condition": volatility_condition,
            "price_at_upper_band": latest['close'] >= latest['bb_upper'],
            "price_at_lower_band": latest['close'] <= latest['bb_lower']
        }
    
    def _analyze_support_resistance(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze support and resistance levels.
        
        Args:
            df: DataFrame with indicators
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary with support/resistance analysis
        """
        # Simple implementation - can be made more sophisticated
        # Get recent highs and lows
        recent_df = df.tail(50)
        
        # Find local maxima and minima
        price_high = recent_df['high'].max()
        price_low = recent_df['low'].min()
        
        # Current price
        current_price = df['close'].iloc[-1]
        
        # Find closest support and resistance
        support_level = recent_df['low'].rolling(window=5).min().dropna().iloc[-1]
        resistance_level = recent_df['high'].rolling(window=5).max().dropna().iloc[-1]
        
        # Distance to support/resistance
        distance_to_support = (current_price - support_level) / current_price
        distance_to_resistance = (resistance_level - current_price) / current_price
        
        return {
            "support_level": support_level,
            "resistance_level": resistance_level,
            "distance_to_support": distance_to_support,
            "distance_to_resistance": distance_to_resistance,
            "price_range": [price_low, price_high],
            "nearest_level": "support" if distance_to_support < distance_to_resistance else "resistance"
        }
    
    def _generate_summary(self, results: Dict[str, Any], df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate a summary of the analysis.
        
        Args:
            results: Results from all analyzers
            df: DataFrame with indicators
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary with summary
        """
        trend = results["trend"]
        momentum = results["momentum"]
        volatility = results["volatility"]
        
        # Simplified trading signal
        signal = "neutral"
        confidence = 0.5
        
        # Strong buy conditions
        if (trend["direction"] in ["bullish", "strong_bullish"] and 
            momentum["rsi_condition"] == "oversold" and
            momentum["macd_direction"] == "bullish" and
            volatility["price_at_lower_band"]):
            signal = "strong_buy"
            confidence = 0.9
        # Buy conditions
        elif (trend["direction"] in ["bullish", "strong_bullish"] and 
              momentum["momentum_strength"] > 0 and
              not volatility["price_at_upper_band"]):
            signal = "buy"
            confidence = 0.7
        # Strong sell conditions
        elif (trend["direction"] in ["bearish", "strong_bearish"] and 
              momentum["rsi_condition"] == "overbought" and
              momentum["macd_direction"] == "bearish" and
              volatility["price_at_upper_band"]):
            signal = "strong_sell"
            confidence = 0.9
        # Sell conditions
        elif (trend["direction"] in ["bearish", "strong_bearish"] and 
              momentum["momentum_strength"] < 0 and
              not volatility["price_at_lower_band"]):
            signal = "sell"
            confidence = 0.7
            
        # Technical analysis commentary
        commentary = self._generate_commentary(signal, results, symbol, timeframe)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "commentary": commentary,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_commentary(self, signal: str, results: Dict[str, Any], symbol: str, timeframe: str) -> str:
        """
        Generate technical analysis commentary.
        
        Args:
            signal: Trading signal
            results: Analysis results
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            
        Returns:
            Commentary text
        """
        trend = results["trend"]
        momentum = results["momentum"]
        volatility = results["volatility"]
        support_resistance = results["support_resistance"]
        
        comments = []
        
        # Trend comments
        if trend["direction"] == "strong_bullish":
            comments.append(f"{symbol} is in a strong uptrend with all moving averages aligned bullishly.")
        elif trend["direction"] == "bullish":
            comments.append(f"{symbol} shows a bullish trend with short-term MA above medium-term MA.")
        elif trend["direction"] == "strong_bearish":
            comments.append(f"{symbol} is in a strong downtrend with all moving averages aligned bearishly.")
        elif trend["direction"] == "bearish":
            comments.append(f"{symbol} shows a bearish trend with short-term MA below medium-term MA.")
        else:
            comments.append(f"{symbol} is in a sideways or neutral trend.")
            
        # Momentum comments
        if momentum["rsi_condition"] == "overbought":
            comments.append(f"RSI at {momentum['rsi_value']:.1f} indicates overbought conditions.")
        elif momentum["rsi_condition"] == "oversold":
            comments.append(f"RSI at {momentum['rsi_value']:.1f} indicates oversold conditions.")
            
        if momentum["macd_signal_crossover"]:
            direction = "bullish" if momentum["macd_histogram"] > 0 else "bearish"
            comments.append(f"MACD shows a recent {direction} crossover.")
            
        # Volatility comments
        if volatility["volatility_condition"] == "high":
            comments.append(f"High volatility detected with Bollinger Band width at {volatility['bb_width']:.3f}.")
        elif volatility["volatility_condition"] == "low":
            comments.append("Low volatility suggests a potential breakout soon.")
            
        if volatility["price_at_upper_band"]:
            comments.append("Price testing upper Bollinger Band, watch for resistance.")
        elif volatility["price_at_lower_band"]:
            comments.append("Price testing lower Bollinger Band, watch for support.")
            
        # Support/Resistance comments
        nearest_level = support_resistance["nearest_level"]
        if nearest_level == "support":
            comments.append(f"Price closer to support at {support_resistance['support_level']:.2f}.")
        else:
            comments.append(f"Price closer to resistance at {support_resistance['resistance_level']:.2f}.")
            
        # Signal-based conclusion
        if signal == "strong_buy":
            comments.append(f"STRONG BUY signal for {symbol} on {timeframe} timeframe with multiple bullish indicators aligned.")
        elif signal == "buy":
            comments.append(f"BUY signal for {symbol} on {timeframe} timeframe with favorable risk-reward.")
        elif signal == "strong_sell":
            comments.append(f"STRONG SELL signal for {symbol} on {timeframe} timeframe with multiple bearish indicators aligned.")
        elif signal == "sell":
            comments.append(f"SELL signal for {symbol} on {timeframe} timeframe with favorable risk-reward.")
        else:
            comments.append(f"NEUTRAL outlook for {symbol} on {timeframe} timeframe. Wait for clearer signals.")
            
        return " ".join(comments) 