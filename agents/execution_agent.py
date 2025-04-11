"""
Execution Agent - Responsible for executing trades in the market.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from agents.base_agent import BaseAgent

class ExecutionAgent(BaseAgent):
    """
    Agent responsible for executing trades in the market.
    
    This agent:
    - Validates trade signals
    - Executes trades
    - Manages order lifecycle
    - Tracks execution performance
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """Initialize the execution agent."""
        super().__init__(name, description, config, parent_id, agent_type="execution")
        
        # Trading configuration
        self.max_slippage = config.get("max_slippage", 0.01)  # 1% max slippage
        self.min_trade_value = config.get("min_trade_value", 10.0)  # Minimum trade value in USD
        self.max_trade_value = config.get("max_trade_value", 1000.0)  # Maximum trade value in USD
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_slippage = 0.0
        self.total_fees = 0.0
        
        self.logger.info(f"Execution agent {name} initialized")
    
    def execute_trade(self, trade_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on a signal.
        
        Args:
            trade_signal: Trade signal dictionary containing:
                - symbol: Trading pair
                - side: 'buy' or 'sell'
                - amount: Trade amount
                - price: Target price
                - type: Order type ('market', 'limit', etc.)
                
        Returns:
            Dictionary containing execution results
        """
        try:
            # Validate trade signal
            if not self._validate_trade_signal(trade_signal):
                raise ValueError("Invalid trade signal")
            
            # In demo mode, simulate trade execution
            execution_result = {
                'success': True,
                'trade_id': f"demo_{datetime.now().timestamp()}",
                'symbol': trade_signal['symbol'],
                'side': trade_signal['side'],
                'amount': trade_signal['amount'],
                'price': trade_signal['price'],
                'timestamp': datetime.now().isoformat(),
                'fees': trade_signal['amount'] * trade_signal['price'] * 0.001,  # 0.1% fee
                'slippage': 0.0
            }
            
            # Update performance metrics
            self._update_metrics(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            self.failed_trades += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_trade_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate a trade signal."""
        required_fields = ['symbol', 'side', 'amount', 'price', 'type']
        
        # Check required fields
        if not all(field in signal for field in required_fields):
            self.logger.error(f"Missing required fields in trade signal: {required_fields}")
            return False
        
        # Validate side
        if signal['side'] not in ['buy', 'sell']:
            self.logger.error(f"Invalid trade side: {signal['side']}")
            return False
        
        # Validate amount
        trade_value = signal['amount'] * signal['price']
        if trade_value < self.min_trade_value:
            self.logger.error(f"Trade value {trade_value} below minimum {self.min_trade_value}")
            return False
        if trade_value > self.max_trade_value:
            self.logger.error(f"Trade value {trade_value} above maximum {self.max_trade_value}")
            return False
        
        return True
    
    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update performance metrics after trade execution."""
        if result['success']:
            self.total_trades += 1
            self.successful_trades += 1
            self.total_fees += result['fees']
            self.total_slippage += result['slippage']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0.0,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'average_slippage': self.total_slippage / self.successful_trades if self.successful_trades > 0 else 0.0
        } 