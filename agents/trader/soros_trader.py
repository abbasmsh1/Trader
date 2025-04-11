"""
George Soros Trader Agent - Reflexivity theory strategy.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.trader.base_trader import BaseTraderAgent

class SorosTraderAgent(BaseTraderAgent):
    """
    Implements George Soros's reflexivity theory.
    
    Key principles:
    - Market prices influence fundamentals
    - Fundamentals influence market prices
    - Feedback loops create trends
    - Market sentiment drives momentum
    - Identify and exploit market inefficiencies
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the Soros trader agent."""
        super().__init__(name, description, config, parent_id)
        
        # Strategy parameters
        self.trend_threshold = config.get("trend_threshold", 0.05)  # 5% trend threshold
        self.sentiment_weight = config.get("sentiment_weight", 0.4)  # 40% weight on sentiment
        self.max_position_size = config.get("max_position_size", 0.3)  # 30% max position size
        
        # Market state tracking
        self.market_trends = {}  # Track market trends
        self.sentiment_scores = {}  # Track market sentiment
        
        self.logger.info(f"Soros trader {name} initialized")
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using reflexivity theory.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        for symbol, data in market_data.items():
            try:
                # Calculate trend strength
                trend_strength = self._calculate_trend_strength(data)
                
                # Calculate market sentiment
                sentiment_score = self._calculate_sentiment(data)
                
                # Calculate reflexivity score
                reflexivity_score = self._calculate_reflexivity_score(
                    trend_strength,
                    sentiment_score,
                    data
                )
                
                # Store analysis
                analysis[symbol] = {
                    'trend_strength': trend_strength,
                    'sentiment_score': sentiment_score,
                    'reflexivity_score': reflexivity_score,
                    'recommendation': self._get_recommendation(reflexivity_score)
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        return analysis
    
    def _calculate_trend_strength(self, data: Dict[str, Any]) -> float:
        """Calculate trend strength using price and volume data."""
        # Use moving averages to identify trend
        ma_20 = data.get('ma_20', data['price'])
        ma_50 = data.get('ma_50', data['price'])
        ma_200 = data.get('ma_200', data['price'])
        
        # Calculate trend indicators
        short_trend = (ma_20 - ma_50) / ma_50
        long_trend = (ma_50 - ma_200) / ma_200
        
        # Combine trends with volume confirmation
        volume_ratio = data.get('volume', 0) / data.get('avg_volume', 1)
        trend_strength = (short_trend + long_trend) * volume_ratio
        
        return trend_strength
    
    def _calculate_sentiment(self, data: Dict[str, Any]) -> float:
        """Calculate market sentiment score."""
        sentiment = 0.0
        
        # Use technical indicators for sentiment
        if 'rsi' in data:
            rsi = data['rsi']
            if rsi < 30:  # Oversold
                sentiment += 0.3
            elif rsi > 70:  # Overbought
                sentiment -= 0.3
        
        # Volume analysis
        volume_ratio = data.get('volume', 0) / data.get('avg_volume', 1)
        if volume_ratio > 1.5:  # High volume
            sentiment += 0.2
        elif volume_ratio < 0.5:  # Low volume
            sentiment -= 0.1
        
        # Volatility
        if 'volatility' in data:
            if data['volatility'] > data.get('avg_volatility', 0):
                sentiment += 0.1
        
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_reflexivity_score(self, 
                                   trend_strength: float, 
                                   sentiment_score: float,
                                   data: Dict[str, Any]) -> float:
        """
        Calculate reflexivity score combining trend and sentiment.
        
        The reflexivity score measures how much the market's perception
        is influencing the underlying fundamentals and vice versa.
        """
        # Base score on trend and sentiment
        base_score = (trend_strength * (1 - self.sentiment_weight) + 
                     sentiment_score * self.sentiment_weight)
        
        # Adjust for momentum
        momentum = data.get('momentum', 0)
        if abs(momentum) > self.trend_threshold:
            base_score *= (1 + momentum)
        
        # Adjust for volume confirmation
        volume_ratio = data.get('volume', 0) / data.get('avg_volume', 1)
        if volume_ratio > 1.5:
            base_score *= 1.2
        elif volume_ratio < 0.5:
            base_score *= 0.8
        
        return base_score
    
    def _get_recommendation(self, reflexivity_score: float) -> str:
        """Get trading recommendation based on reflexivity score."""
        if reflexivity_score > self.trend_threshold:
            return 'buy'
        elif reflexivity_score < -self.trend_threshold:
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
                    # Calculate position size based on reflexivity score
                    position_size = self._calculate_position_size(data)
                    
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'market',  # Soros often uses market orders to capture momentum
                        'amount': position_size,
                        'reason': f"Strong reflexivity signal: {data['reflexivity_score']:.2f}"
                    })
                    
                elif recommendation == 'sell' and symbol in self.positions:
                    signals.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'market',
                        'amount': self.positions[symbol]['amount'],
                        'reason': f"Negative reflexivity signal: {data['reflexivity_score']:.2f}"
                    })
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        return signals
    
    def _calculate_position_size(self, analysis: Dict[str, Any]) -> float:
        """Calculate position size based on reflexivity score and risk parameters."""
        # Base size on reflexivity score
        reflexivity_score = analysis['reflexivity_score']
        size_factor = min(abs(reflexivity_score) / self.trend_threshold, 1.0)
        
        # Apply maximum position size constraint
        max_amount = self.portfolio_value * self.max_position_size
        position_size = max_amount * size_factor
        
        return position_size
    
    def update_portfolio(self, trade_result: Dict[str, Any]) -> None:
        """Update portfolio after trade execution."""
        super().update_portfolio(trade_result)
        
        # Update market state tracking
        if trade_result['success']:
            symbol = trade_result['symbol']
            if trade_result['side'] == 'buy':
                self.market_trends[symbol] = {
                    'entry_price': trade_result['price'],
                    'entry_time': trade_result['timestamp'],
                    'sentiment': self.sentiment_scores.get(symbol, 0)
                }
            else:  # sell
                if symbol in self.market_trends:
                    del self.market_trends[symbol] 