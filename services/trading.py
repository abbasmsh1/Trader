"""
Trading Service - Handles trade execution for traders.
"""

import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass

@dataclass
class TradeSignal:
    """Container for trade signals."""
    trader_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market' or 'limit'
    amount: Decimal
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    timestamp: datetime = datetime.now()

class TradingService:
    """Service for executing trades based on trader signals."""
    
    def __init__(self, trade_execution_service: Any, performance_tracker: Any, db_handler: Any):
        """
        Initialize the trading service.
        
        Args:
            trade_execution_service: TradeExecutionService instance
            performance_tracker: PerformanceTracker instance
            db_handler: Database handler instance
        """
        self.trade_execution = trade_execution_service
        self.performance_tracker = performance_tracker
        self.db = db_handler
        self.logger = logging.getLogger('trading_service')
        self.logger.info("Trading Service initialized")
    
    def process_trade_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """
        Process a trade signal and execute the trade.
        
        Args:
            signal: TradeSignal object containing trade details
            
        Returns:
            Dict containing trade execution results
        """
        try:
            # Validate signal
            self._validate_signal(signal)
            
            # Check trader's balance
            balance = self._check_trader_balance(signal)
            if not balance:
                return {
                    'success': False,
                    'error': 'Insufficient balance'
                }
            
            # Execute the trade
            order_response = self._execute_trade(signal)
            
            # Record the trade
            trade_record = self._record_trade(signal, order_response)
            
            # Update performance metrics
            self._update_performance(signal.trader_id)
            
            return {
                'success': True,
                'order': order_response,
                'trade_record': trade_record
            }
            
        except Exception as e:
            self.logger.error(f"Error processing trade signal: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_signal(self, signal: TradeSignal) -> None:
        """Validate a trade signal."""
        if not signal.trader_id or not signal.symbol or not signal.side or not signal.amount:
            raise ValueError("Missing required trade signal parameters")
        
        if signal.side not in ['buy', 'sell']:
            raise ValueError("Invalid trade side")
        
        if signal.order_type not in ['market', 'limit']:
            raise ValueError("Invalid order type")
        
        if signal.order_type == 'limit' and not signal.price:
            raise ValueError("Price required for limit orders")
    
    def _check_trader_balance(self, signal: TradeSignal) -> bool:
        """Check if trader has sufficient balance for the trade."""
        try:
            # Get trader's wallet balance
            balance = self.db.get_trader_balance(signal.trader_id)
            if not balance:
                return False
            
            # Calculate required amount
            required_amount = signal.amount
            if signal.side == 'buy':
                # For buy orders, check quote currency balance
                quote_currency = signal.symbol.split('/')[1]
                required_amount = signal.amount * (signal.price or self._get_current_price(signal.symbol))
            
            # Check if balance is sufficient
            return balance >= required_amount
            
        except Exception as e:
            self.logger.error(f"Error checking balance: {str(e)}")
            return False
    
    def _execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade on the exchange."""
        try:
            # Place the main order
            order_params = {
                'symbol': signal.symbol,
                'side': signal.side,
                'order_type': signal.order_type,
                'amount': signal.amount,
                'price': signal.price
            }
            
            order_response = self.trade_execution.place_order(**order_params)
            
            # Place stop loss if specified
            if signal.stop_loss:
                stop_loss_params = {
                    'symbol': signal.symbol,
                    'side': 'sell' if signal.side == 'buy' else 'buy',
                    'order_type': 'stop',
                    'amount': signal.amount,
                    'price': signal.stop_loss,
                    'params': {'stopPrice': signal.stop_loss}
                }
                self.trade_execution.place_order(**stop_loss_params)
            
            # Place take profit if specified
            if signal.take_profit:
                take_profit_params = {
                    'symbol': signal.symbol,
                    'side': 'sell' if signal.side == 'buy' else 'buy',
                    'order_type': 'limit',
                    'amount': signal.amount,
                    'price': signal.take_profit
                }
                self.trade_execution.place_order(**take_profit_params)
            
            return order_response
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise
    
    def _record_trade(self, signal: TradeSignal, order_response: Dict[str, Any]) -> Dict[str, Any]:
        """Record a trade in the database."""
        try:
            trade_record = {
                'trader_id': signal.trader_id,
                'symbol': signal.symbol,
                'side': signal.side,
                'order_type': signal.order_type,
                'amount': float(signal.amount),
                'price': float(signal.price) if signal.price else None,
                'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                'take_profit': float(signal.take_profit) if signal.take_profit else None,
                'order_id': order_response['id'],
                'status': order_response['status'],
                'timestamp': signal.timestamp.isoformat(),
                'execution_time': datetime.now().isoformat()
            }
            
            self.db.record_trade(trade_record)
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
            raise
    
    def _update_performance(self, trader_id: str) -> None:
        """Update trader's performance metrics."""
        try:
            self.performance_tracker.calculate_metrics(trader_id)
        except Exception as e:
            self.logger.error(f"Error updating performance: {str(e)}")
    
    def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol."""
        try:
            ticker = self.trade_execution.get_ticker(symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            raise 