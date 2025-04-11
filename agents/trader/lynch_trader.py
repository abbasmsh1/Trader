"""
Peter Lynch Trader Agent - Growth investing strategy.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from agents.trader.base_trader import BaseTraderAgent

class LynchTraderAgent(BaseTraderAgent):
    """
    Implements Peter Lynch's growth investing strategy.
    
    Key principles:
    - Growth at a reasonable price (GARP)
    - Focus on fundamentals
    - Long-term perspective
    - Understand what you own
    - Look for hidden value
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the Lynch trader agent."""
        super().__init__(name, description, config, parent_id)
        
        # Strategy parameters
        self.min_growth_rate = config.get("min_growth_rate", 0.15)  # 15% minimum growth
        self.max_pe_ratio = config.get("max_pe_ratio", 25)  # Maximum P/E ratio
        self.min_peg_ratio = config.get("min_peg_ratio", 0.5)  # Minimum PEG ratio
        self.max_peg_ratio = config.get("max_peg_ratio", 1.5)  # Maximum PEG ratio
        self.max_position_size = config.get("max_position_size", 0.15)  # 15% max position size
        
        # Portfolio tracking
        self.holdings = {}  # Current holdings with analysis
        self.watchlist = {}  # Potential investments
        
        self.logger.info(f"Lynch trader {name} initialized")
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data using growth investing principles.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        for symbol, data in market_data.items():
            try:
                # Calculate growth metrics
                growth_metrics = self._calculate_growth_metrics(data)
                
                # Calculate valuation metrics
                valuation_metrics = self._calculate_valuation_metrics(data)
                
                # Calculate quality metrics
                quality_metrics = self._calculate_quality_metrics(data)
                
                # Combine metrics into overall score
                score = self._calculate_lynch_score(
                    growth_metrics,
                    valuation_metrics,
                    quality_metrics
                )
                
                # Store analysis
                analysis[symbol] = {
                    'growth_metrics': growth_metrics,
                    'valuation_metrics': valuation_metrics,
                    'quality_metrics': quality_metrics,
                    'lynch_score': score,
                    'recommendation': self._get_recommendation(score)
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        return analysis
    
    def _calculate_growth_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate growth-related metrics."""
        metrics = {}
        
        # Revenue growth
        revenue_growth = data.get('revenue_growth', 0)
        metrics['revenue_growth'] = revenue_growth
        
        # Earnings growth
        earnings_growth = data.get('earnings_growth', 0)
        metrics['earnings_growth'] = earnings_growth
        
        # Cash flow growth
        cash_flow_growth = data.get('cash_flow_growth', 0)
        metrics['cash_flow_growth'] = cash_flow_growth
        
        # Market share growth
        market_share_growth = data.get('market_share_growth', 0)
        metrics['market_share_growth'] = market_share_growth
        
        return metrics
    
    def _calculate_valuation_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate valuation metrics."""
        metrics = {}
        
        # P/E ratio
        pe_ratio = data.get('pe_ratio', float('inf'))
        metrics['pe_ratio'] = pe_ratio
        
        # PEG ratio
        earnings_growth = data.get('earnings_growth', 0)
        if earnings_growth > 0:
            peg_ratio = pe_ratio / earnings_growth
        else:
            peg_ratio = float('inf')
        metrics['peg_ratio'] = peg_ratio
        
        # Price to book ratio
        price_to_book = data.get('price_to_book', float('inf'))
        metrics['price_to_book'] = price_to_book
        
        # Price to sales ratio
        price_to_sales = data.get('price_to_sales', float('inf'))
        metrics['price_to_sales'] = price_to_sales
        
        return metrics
    
    def _calculate_quality_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate business quality metrics."""
        metrics = {}
        
        # Return on equity
        roe = data.get('return_on_equity', 0)
        metrics['return_on_equity'] = roe
        
        # Debt to equity ratio
        debt_to_equity = data.get('debt_to_equity', float('inf'))
        metrics['debt_to_equity'] = debt_to_equity
        
        # Operating margin
        operating_margin = data.get('operating_margin', 0)
        metrics['operating_margin'] = operating_margin
        
        # Free cash flow margin
        fcf_margin = data.get('fcf_margin', 0)
        metrics['fcf_margin'] = fcf_margin
        
        return metrics
    
    def _calculate_lynch_score(self,
                             growth_metrics: Dict[str, float],
                             valuation_metrics: Dict[str, float],
                             quality_metrics: Dict[str, float]) -> float:
        """
        Calculate overall Lynch score.
        
        Higher score indicates better investment opportunity.
        """
        score = 0.0
        
        # Growth score (40% weight)
        growth_score = (
            growth_metrics['revenue_growth'] * 0.3 +
            growth_metrics['earnings_growth'] * 0.4 +
            growth_metrics['cash_flow_growth'] * 0.2 +
            growth_metrics['market_share_growth'] * 0.1
        )
        
        # Valuation score (30% weight)
        valuation_score = 0.0
        if valuation_metrics['pe_ratio'] <= self.max_pe_ratio:
            valuation_score += 0.4
        if self.min_peg_ratio <= valuation_metrics['peg_ratio'] <= self.max_peg_ratio:
            valuation_score += 0.6
        
        # Quality score (30% weight)
        quality_score = (
            min(quality_metrics['return_on_equity'] / 20.0, 1.0) * 0.4 +
            min(5.0 / quality_metrics['debt_to_equity'], 1.0) * 0.3 +
            min(quality_metrics['operating_margin'] / 0.2, 1.0) * 0.3
        )
        
        # Combine scores
        score = (
            growth_score * 0.4 +
            valuation_score * 0.3 +
            quality_score * 0.3
        )
        
        return score
    
    def _get_recommendation(self, lynch_score: float) -> str:
        """Get trading recommendation based on Lynch score."""
        if lynch_score >= 0.8:  # Strong buy
            return 'buy'
        elif lynch_score <= 0.2:  # Strong sell
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
                    # Calculate position size based on Lynch score
                    position_size = self._calculate_position_size(data)
                    
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'limit',
                        'amount': position_size,
                        'reason': (f"Lynch score: {data['lynch_score']:.2f}, "
                                 f"Growth: {data['growth_metrics']['earnings_growth']:.1%}")
                    })
                    
                elif recommendation == 'sell' and symbol in self.positions:
                    signals.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'limit',
                        'amount': self.positions[symbol]['amount'],
                        'reason': (f"Lynch score: {data['lynch_score']:.2f}, "
                                 f"PE ratio: {data['valuation_metrics']['pe_ratio']:.1f}")
                    })
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
        
        return signals
    
    def _calculate_position_size(self, analysis: Dict[str, Any]) -> float:
        """Calculate position size based on Lynch score and growth metrics."""
        # Base size on Lynch score
        lynch_score = analysis['lynch_score']
        size_factor = min(lynch_score, 1.0)
        
        # Adjust for growth rate
        growth_rate = analysis['growth_metrics']['earnings_growth']
        if growth_rate > self.min_growth_rate:
            size_factor *= min(growth_rate / self.min_growth_rate, 1.5)
        
        # Apply maximum position size constraint
        max_amount = self.portfolio_value * self.max_position_size
        position_size = max_amount * size_factor
        
        return position_size
    
    def update_portfolio(self, trade_result: Dict[str, Any]) -> None:
        """Update portfolio after trade execution."""
        super().update_portfolio(trade_result)
        
        # Update holdings tracking
        if trade_result['success']:
            symbol = trade_result['symbol']
            if trade_result['side'] == 'buy':
                self.holdings[symbol] = {
                    'entry_price': trade_result['price'],
                    'entry_time': trade_result['timestamp'],
                    'initial_analysis': self.watchlist.get(symbol, {})
                }
                if symbol in self.watchlist:
                    del self.watchlist[symbol]
            else:  # sell
                if symbol in self.holdings:
                    del self.holdings[symbol] 