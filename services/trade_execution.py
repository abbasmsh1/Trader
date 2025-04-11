"""
Trade Execution Service - Handles order placement and management.
"""

import logging
import ccxt
from typing import Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
import os

class TradeExecutionService:
    """Service for executing trades and managing orders."""
    
    def __init__(self, config: Dict[str, Any], testnet: bool = True):
        """
        Initialize the trade execution service.
        
        Args:
            config: Configuration dictionary
            testnet: Whether to use testnet (default: True)
        """
        self.logger = logging.getLogger('trade_execution')
        self.config = config
        self.testnet = testnet
        self.exchange = self._setup_exchange()
        self.logger.info("Trade Execution Service initialized")
    
    def _setup_exchange(self) -> ccxt.Exchange:
        """Set up the exchange connection."""
        try:
            # Initialize exchange
            exchange_class = getattr(ccxt, self.config['exchange']['name'].lower())
            exchange = exchange_class({
                'apiKey': os.getenv('EXCHANGE_API_KEY'),
                'secret': os.getenv('EXCHANGE_API_SECRET'),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'testnet': self.testnet
                }
            })
            
            # Test connection
            exchange.load_markets()
            self.logger.info(f"Connected to {exchange.name} exchange")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Error setting up exchange: {str(e)}")
            raise
    
    def place_order(self, symbol: str, side: str, order_type: str,
                   amount: Decimal, price: Optional[Decimal] = None,
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market' or 'limit')
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional order parameters
            
        Returns:
            Order response from exchange
        """
        try:
            # Validate inputs
            if order_type == 'limit' and price is None:
                raise ValueError("Price is required for limit orders")
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': float(amount),
                'params': params or {}
            }
            
            if price is not None:
                order_params['price'] = float(price)
            
            # Place order
            response = self.exchange.create_order(**order_params)
            
            self.logger.info(f"Order placed: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Trading pair symbol
            
        Returns:
            Cancellation response from exchange
        """
        try:
            response = self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order cancelled: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            raise
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order
            symbol: Trading pair symbol
            
        Returns:
            Order status from exchange
        """
        try:
            response = self.exchange.fetch_order(order_id, symbol)
            self.logger.info(f"Order status: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")
            raise
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """
        Get all open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter orders
            
        Returns:
            List of open orders
        """
        try:
            response = self.exchange.fetch_open_orders(symbol)
            self.logger.info(f"Open orders: {len(response)}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {str(e)}")
            raise
    
    def get_balance(self, currency: str) -> Dict[str, Any]:
        """
        Get balance for a specific currency.
        
        Args:
            currency: Currency symbol (e.g., 'USDT')
            
        Returns:
            Balance information
        """
        try:
            balance = self.exchange.fetch_balance()
            return {
                'free': Decimal(str(balance[currency]['free'])),
                'used': Decimal(str(balance[currency]['used'])),
                'total': Decimal(str(balance[currency]['total']))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting balance: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': Decimal(str(ticker['last'])),
                'bid': Decimal(str(ticker['bid'])),
                'ask': Decimal(str(ticker['ask'])),
                'volume': Decimal(str(ticker['baseVolume'])),
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ticker: {str(e)}")
            raise 