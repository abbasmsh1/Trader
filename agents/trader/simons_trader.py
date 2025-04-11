"""
Jim Simons Trader Agent - Quantitative trading strategy.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.trader.base_trader import BaseTraderAgent

class SimonsTraderAgent(BaseTraderAgent):
    """
    Implements Jim Simons's quantitative trading strategy.
    
    Key principles:
    - Statistical arbitrage
    - High-frequency pattern recognition
    - Mathematical models
    - Risk management through diversification
    - Market-neutral approach
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the Simons trader agent."""
        super().__init__(name, description, config, parent_id)
        
        # Strategy parameters
        self.lookback_period = config.get("lookback_period", 100)  # Data points for analysis
        self.signal_threshold = config.get("signal_threshold", 2.0)  # Z-score threshold
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% max position size
        self.min_data_points = config.get("min_data_points", 30)  # Minimum data points needed
        
        # Model state
        self.price_history = {}  # Historical price data
        self.volatility_history = {}  # Historical volatility
        self.correlation_matrix = None  # Asset correlations
        
        self.logger.info(f"Simons trader {name} initialized")
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using quantitative methods.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        # Update historical data
        self._update_history(market_data)
        
        # Calculate correlations if enough data
        if self._has_sufficient_data():
            self._update_correlation_matrix()
        
        for symbol, data in market_data.items():
            try:
                # Statistical analysis
                z_score = self._calculate_z_score(symbol, data)
                volatility = self._calculate_volatility(symbol)
                momentum = self._calculate_momentum(symbol)
                
                # Generate signals based on statistical analysis
                signals = self._analyze_patterns(symbol, z_score, volatility, momentum)
                
                # Store analysis
                analysis[symbol] = {
                    'z_score': z_score,
                    'volatility': volatility,
                    'momentum': momentum,
                    'signals': signals,
                    'recommendation': self._get_recommendation(signals)
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        return analysis
    
    def _update_history(self, market_data: Dict[str, Any]) -> None:
        """Update price and volatility history."""
        for symbol, data in market_data.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            
            # Add new price data
            self.price_history[symbol].append(data['price'])
            if len(self.price_history[symbol]) > self.lookback_period:
                self.price_history[symbol].pop(0)
            
            # Calculate and store volatility
            if len(self.price_history[symbol]) > 1:
                returns = np.diff(self.price_history[symbol]) / self.price_history[symbol][:-1]
                volatility = np.std(returns)
                self.volatility_history[symbol].append(volatility)
                if len(self.volatility_history[symbol]) > self.lookback_period:
                    self.volatility_history[symbol].pop(0)
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data for analysis."""
        return all(len(hist) >= self.min_data_points 
                  for hist in self.price_history.values())
    
    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix between assets."""
        symbols = list(self.price_history.keys())
        n_assets = len(symbols)
        correlations = np.zeros((n_assets, n_assets))
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlations[i,j] = 1.0
                else:
                    # Calculate correlation using returns
                    returns1 = np.diff(self.price_history[symbol1])
                    returns2 = np.diff(self.price_history[symbol2])
                    correlation = np.corrcoef(returns1, returns2)[0,1]
                    correlations[i,j] = correlation
        
        self.correlation_matrix = correlations
    
    def _calculate_z_score(self, symbol: str, data: Dict[str, Any]) -> float:
        """Calculate z-score for current price."""
        if len(self.price_history[symbol]) < self.min_data_points:
            return 0.0
        
        prices = np.array(self.price_history[symbol])
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return 0.0
            
        return (data['price'] - mean_price) / std_price
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility."""
        if len(self.volatility_history[symbol]) < 2:
            return 0.0
        
        return self.volatility_history[symbol][-1]
    
    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate price momentum."""
        if len(self.price_history[symbol]) < self.min_data_points:
            return 0.0
        
        prices = np.array(self.price_history[symbol])
        returns = np.diff(prices) / prices[:-1]
        
        # Use exponential weighted average for momentum
        weights = np.exp(np.linspace(-1., 0., len(returns)))
        weights /= weights.sum()
        
        return np.sum(returns * weights)
    
    def _analyze_patterns(self, 
                         symbol: str, 
                         z_score: float, 
                         volatility: float,
                         momentum: float) -> Dict[str, float]:
        """
        Analyze trading patterns using multiple factors.
        
        Returns a dictionary of signal strengths for different patterns.
        """
        signals = {
            'mean_reversion': -z_score,  # High z-score suggests overvalued
            'momentum': momentum,
            'volatility': 1.0 - volatility  # Lower volatility is better
        }
        
        # Adjust for correlation effects if available
        if self.correlation_matrix is not None:
            symbols = list(self.price_history.keys())
            idx = symbols.index(symbol)
            
            # Calculate average correlation with other assets
            correlations = self.correlation_matrix[idx]
            avg_correlation = np.mean(correlations[correlations != 1.0])
            
            # Reduce signal strength for highly correlated assets
            correlation_factor = 1.0 - abs(avg_correlation)
            for key in signals:
                signals[key] *= correlation_factor
        
        return signals
    
    def _get_recommendation(self, signals: Dict[str, float]) -> str:
        """Get trading recommendation based on combined signals."""
        # Combine signals with equal weights
        combined_signal = sum(signals.values()) / len(signals)
        
        if combined_signal > self.signal_threshold:
            return 'buy'
        elif combined_signal < -self.signal_threshold:
            return 'sell'
        return 'hold'
    
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on analysis.
        
        Args:
            analysis: Market analysis results
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for symbol, data in analysis.items():
            try:
                recommendation = data['recommendation']
                
                if recommendation == 'buy' and symbol not in self.positions:
                    # Calculate position size based on signals
                    position_size = self._calculate_position_size(data)
                    
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'limit',  # Use limit orders for better execution
                        'amount': position_size,
                        'reason': (f"Z-score: {data['z_score']:.2f}, "
                                 f"Momentum: {data['signals']['momentum']:.2f}")
                    })
                    
                elif recommendation == 'sell' and symbol in self.positions:
                    signals.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'limit',
                        'amount': self.positions[symbol]['amount'],
                        'reason': (f"Z-score: {data['z_score']:.2f}, "
                                 f"Momentum: {data['signals']['momentum']:.2f}")
                    })
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        return signals
    
    def _calculate_position_size(self, analysis: Dict[str, Any]) -> float:
        """Calculate position size based on signal strength and volatility."""
        # Base size on signal strength
        signal_strength = abs(analysis['z_score'])
        size_factor = min(signal_strength / self.signal_threshold, 1.0)
        
        # Adjust for volatility
        volatility_factor = 1.0 - min(analysis['volatility'], 0.5)  # Cap volatility adjustment
        size_factor *= volatility_factor
        
        # Apply maximum position size constraint
        max_amount = self.portfolio_value * self.max_position_size
        position_size = max_amount * size_factor
        
        return position_size 