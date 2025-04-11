"""
Warren Buffett Trader Agent - Value investing strategy.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.trader.base_trader import BaseTraderAgent

class BuffettTraderAgent(BaseTraderAgent):
    """
    Implements Warren Buffett's value investing strategy.
    
    Key principles:
    - Focus on intrinsic value
    - Long-term investment horizon
    - Margin of safety
    - Quality over price
    - Circle of competence
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the Buffett trader agent."""
        super().__init__(name, description, config, parent_id)
        
        # Strategy parameters
        self.min_margin_of_safety = config.get("min_margin_of_safety", 0.3)  # 30% minimum margin of safety
        self.min_holding_period = config.get("min_holding_period", 30)  # 30 days minimum holding
        self.max_position_size = config.get("max_position_size", 0.2)  # 20% max position size
        
        # Portfolio tracking
        self.positions = {}  # Current positions
        self.watchlist = {}  # Potential investments
        
        self.logger.info(f"Buffett trader {name} initialized")
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using value investing principles.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        for symbol, data in market_data.items():
            try:
                # Calculate intrinsic value
                intrinsic_value = self._calculate_intrinsic_value(data)
                
                # Calculate margin of safety
                current_price = data['price']
                margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
                
                # Store analysis
                analysis[symbol] = {
                    'intrinsic_value': intrinsic_value,
                    'current_price': current_price,
                    'margin_of_safety': margin_of_safety,
                    'recommendation': self._get_recommendation(margin_of_safety)
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        return analysis
    
    def _calculate_intrinsic_value(self, data: Dict[str, Any]) -> float:
        """
        Calculate intrinsic value using fundamental data.
        
        This is a simplified implementation. In reality, would use:
        - Discounted Cash Flow (DCF) analysis
        - Asset valuation
        - Comparable company analysis
        - Growth projections
        """
        # Simplified calculation using moving averages
        ma_50 = data.get('ma_50', data['price'])
        ma_200 = data.get('ma_200', data['price'])
        
        # Add premium for strong fundamentals
        fundamental_premium = 1.0
        if data.get('volume', 0) > data.get('avg_volume', 0):
            fundamental_premium += 0.1
        if data.get('rsi', 50) < 30:  # Oversold
            fundamental_premium += 0.1
        
        # Calculate intrinsic value
        intrinsic_value = ((ma_50 + ma_200) / 2) * fundamental_premium
        
        return intrinsic_value
    
    def _get_recommendation(self, margin_of_safety: float) -> str:
        """Get trading recommendation based on margin of safety."""
        if margin_of_safety >= self.min_margin_of_safety:
            return 'buy'
        elif margin_of_safety <= -self.min_margin_of_safety:
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
                    # Calculate position size
                    position_size = self._calculate_position_size(data)
                    
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'limit',
                        'price': data['current_price'],
                        'amount': position_size,
                        'reason': f"Margin of safety: {data['margin_of_safety']:.2%}"
                    })
                    
                elif recommendation == 'sell' and symbol in self.positions:
                    signals.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'limit',
                        'price': data['current_price'],
                        'amount': self.positions[symbol]['amount'],
                        'reason': f"Margin of safety eroded: {data['margin_of_safety']:.2%}"
                    })
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        return signals
    
    def _calculate_position_size(self, analysis: Dict[str, Any]) -> float:
        """Calculate position size based on margin of safety and risk parameters."""
        # Base size on margin of safety
        margin_of_safety = analysis['margin_of_safety']
        size_factor = min(margin_of_safety / self.min_margin_of_safety, 1.0)
        
        # Apply maximum position size constraint
        max_amount = self.portfolio_value * self.max_position_size
        position_size = max_amount * size_factor
        
        return position_size
    
    def update_portfolio(self, trade_result: Dict[str, Any]) -> None:
        """Update portfolio after trade execution."""
        if trade_result['success']:
            symbol = trade_result['symbol']
            if trade_result['side'] == 'buy':
                self.positions[symbol] = {
                    'amount': trade_result['amount'],
                    'price': trade_result['price'],
                    'timestamp': trade_result['timestamp']
                }
            else:  # sell
                if symbol in self.positions:
                    del self.positions[symbol] 