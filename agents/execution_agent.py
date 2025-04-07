"""
Execution Agent - Handles the execution of trades on exchanges.

This agent is responsible for connecting to exchanges and executing trades
based on the instructions from the portfolio manager.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from agents.base_agent import BaseAgent
from models.wallet import Wallet

class ExecutionAgentError(Exception):
    """Exception raised when trade execution fails."""
    pass

class ExecutionAgent(BaseAgent):
    """
    Execution Agent that handles the connection to exchanges and order execution.
    
    This agent is responsible for:
    - Connecting to cryptocurrency exchanges
    - Executing buy and sell orders
    - Tracking order statuses
    - Reporting execution results
    """
    
    def __init__(self, agent_id: str, name: str, description: str, config: Dict[str, Any], 
                 parent_id: Optional[str] = None, exchange_api=None, wallet: Optional[Wallet] = None):
        """Initialize the execution agent."""
        super().__init__(agent_id, name, description, "execution", config, parent_id)
        
        # Exchange configuration
        self.exchange_name = self.config.get("exchange", "binance")
        self.use_testnet = self.config.get("use_testnet", True)
        self.slippage_model = self.config.get("slippage_model", "conservative")
        self.max_slippage_pct = self.config.get("max_slippage_pct", 1.0)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 5)
        
        # Execution strategies
        self.execution_strategies = self.config.get("execution_strategies", {
            "market": True,
            "limit": True,
            "twap": False,
            "iceberg": False
        })
        
        # Order tracking
        self.order_history = []
        self.active_orders = {}
        self.order_lifetime = self.config.get("order_lifetime", 300)  # seconds
        
        # Performance metrics
        self.metrics = {
            "orders_executed": 0,
            "orders_failed": 0,
            "market_orders": 0,
            "limit_orders": 0,
            "twap_orders": 0,
            "iceberg_orders": 0,
            "avg_slippage": 0,
            "total_fees": 0
        }
        
        # Set up exchange API
        self.exchange_api = exchange_api
        if not self.exchange_api:
            self._setup_exchange_api()
            
        # Set up wallet
        self.wallet = wallet
        if not self.wallet:
            self.wallet = Wallet(
                initial_balance=self.config.get("initial_balance", 10000.0),
                base_currency=self.config.get("base_currency", "USDT"),
                name=f"Execution-{agent_id}"
            )
        
        self.logger = logging.getLogger(f"execution_{agent_id}")
        self.logger.info(f"Initialized execution agent for {self.exchange_name}")
    
    def _setup_exchange_api(self):
        """Set up connection to exchange API."""
        try:
            # In a real implementation, this would use ccxt or another library
            # to connect to the actual exchange API
            # For now, use a dummy implementation
            self.exchange_api = DummyExchangeAPI(self.exchange_name, self.use_testnet)
            self.logger.info(f"Connected to {self.exchange_name} API")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to exchange API: {str(e)}")
            self.exchange_api = None
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute orders based on input data.
        
        Args:
            data: Dictionary containing orders to execute
                 Expected format:
                 {
                     "orders": [
                         {
                             "symbol": "BTC/USDT",
                             "side": "buy",
                             "type": "market",
                             "quantity": 0.1,
                             "price": 50000,  # Optional for market orders
                             "params": {}     # Additional params specific to order type
                         },
                         ...
                     ],
                     "market_data": {...}  # Market data for slippage calculation
                 }
        
        Returns:
            Dictionary with execution results
        """
        if not self.is_active:
            return {"error": "Agent is not active"}
        
        if not self.exchange_api:
            return {"error": "No exchange API available"}
        
        # Extract orders and market data
        orders = data.get("orders", [])
        market_data = data.get("market_data", {})
        
        if not orders:
            return {"message": "No orders to execute", "executed": [], "failed": []}
        
        executed_orders = []
        failed_orders = []
        
        # Process each order
        for order in orders:
            self.logger.info(f"Processing order: {order['side']} {order.get('quantity', 0)} {order.get('symbol', 'unknown')}")
            
            try:
                result = self._execute_order(order, market_data)
                if result.get("success", False):
                    executed_orders.append(result)
                else:
                    failed_orders.append({
                        "order": order,
                        "error": result.get("error", "Unknown execution error")
                    })
            except Exception as e:
                self.logger.error(f"Error executing order: {str(e)}")
                failed_orders.append({
                    "order": order,
                    "error": str(e)
                })
        
        # Update metrics
        self._update_metrics(executed_orders)
        
        # Prepare response
        response = {
            "executed": executed_orders,
            "failed": failed_orders,
            "success_rate": len(executed_orders) / max(1, len(orders)) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Execution complete: {len(executed_orders)} succeeded, {len(failed_orders)} failed")
        
        return response
    
    def _execute_order(self, order: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single order.
        
        Args:
            order: Order details
            market_data: Current market data
            
        Returns:
            Dictionary with execution result
        """
        if not self.exchange_api:
            raise ExecutionAgentError("No exchange API available")
            
        # Extract order details
        symbol = order.get("symbol")
        side = order.get("side")
        order_type = order.get("type", "market")
        quantity = order.get("quantity")
        price = order.get("price")
        params = order.get("params", {})
        
        # Basic validation
        if not symbol or not side or not quantity or quantity <= 0:
            return {
                "success": False,
                "order": order,
                "error": "Invalid order parameters"
            }
            
        # Apply slippage model to price if needed
        adjusted_price = self._apply_slippage(symbol, side, price)
        
        # Select execution strategy based on order type
        execution_result = None
        retry_count = 0
        
        while retry_count < self.retry_attempts and not execution_result:
            try:
                if order_type == "market":
                    execution_result = self._execute_market_order(symbol, side, quantity, params)
                elif order_type == "limit" and adjusted_price:
                    execution_result = self._execute_limit_order(symbol, side, quantity, adjusted_price, params)
                elif order_type == "twap" and adjusted_price:
                    execution_result = self._execute_twap_order(symbol, side, quantity, adjusted_price, params)
                elif order_type == "iceberg" and adjusted_price:
                    execution_result = self._execute_iceberg_order(symbol, side, quantity, adjusted_price, params)
                else:
                    return {
                        "success": False,
                        "order": order,
                        "error": f"Unsupported order type: {order_type}"
                    }
            except Exception as e:
                self.logger.warning(f"Execution attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                if retry_count < self.retry_attempts:
                    time.sleep(self.retry_delay)
        
        if not execution_result:
            return {
                "success": False,
                "order": order,
                "error": "Maximum retry attempts reached"
            }
            
        # Update wallet with the executed trade
        self._update_wallet(execution_result, symbol, side, quantity, execution_result.get("price", adjusted_price))
        
        # Record order in history
        execution_result["timestamp"] = datetime.now().isoformat()
        execution_result["order"] = order
        self.order_history.append(execution_result)
        
        # If order is active (e.g., open limit order), track it
        if execution_result.get("status") == "open":
            self.active_orders[execution_result["id"]] = {
                "order": order,
                "execution": execution_result,
                "created_at": time.time()
            }
            
        return {
            "success": True,
            "execution": execution_result,
            "original_order": order
        }
    
    def _update_wallet(self, execution_result: Dict[str, Any], symbol: str, side: str, quantity: float, price: float) -> None:
        """
        Update wallet with executed trade.
        
        Args:
            execution_result: The result from the exchange
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Trade side ('buy' or 'sell')
            quantity: Amount of base asset
            price: Execution price
        """
        if not self.wallet:
            self.logger.warning("No wallet available to update")
            return
            
        try:
            # Parse symbol to get base/quote currencies
            if '/' in symbol:
                base_currency, quote_currency = symbol.split('/')
            else:
                # Default to common format if not explicitly provided
                base_currency = symbol
                quote_currency = self.wallet.base_currency
            
            # Calculate amounts
            quote_amount = quantity * price
            fee = execution_result.get("fee", {}).get("cost", 0)
            fee_currency = execution_result.get("fee", {}).get("currency", quote_currency if side == "buy" else base_currency)
            
            # Record trade in wallet
            if side == "buy":
                self.wallet.add_trade(
                    trade_type="buy",
                    from_currency=quote_currency,
                    to_currency=base_currency,
                    from_amount=quote_amount,
                    to_amount=quantity,
                    price=price,
                    fee=fee,
                    fee_currency=fee_currency,
                    exchange=self.exchange_name,
                    external_id=execution_result.get("id")
                )
                self.logger.info(f"Wallet updated: Bought {quantity} {base_currency} for {quote_amount} {quote_currency}")
            else:  # sell
                self.wallet.add_trade(
                    trade_type="sell",
                    from_currency=base_currency,
                    to_currency=quote_currency,
                    from_amount=quantity,
                    to_amount=quote_amount,
                    price=price,
                    fee=fee,
                    fee_currency=fee_currency,
                    exchange=self.exchange_name,
                    external_id=execution_result.get("id")
                )
                self.logger.info(f"Wallet updated: Sold {quantity} {base_currency} for {quote_amount} {quote_currency}")
                
        except Exception as e:
            self.logger.error(f"Error updating wallet: {str(e)}")
    
    def _execute_market_order(self, symbol: str, side: str, quantity: float, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a market order."""
        self.logger.info(f"Executing market order: {side} {quantity} {symbol}")
        
        # In a real implementation, this would call the exchange API
        result = self.exchange_api.create_order(
            symbol=symbol,
            side=side,
            order_type="market",
            quantity=quantity
        )
        
        self.metrics["market_orders"] += 1
        return result
    
    def _execute_limit_order(self, symbol: str, side: str, quantity: float, price: float, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a limit order."""
        self.logger.info(f"Executing limit order: {side} {quantity} {symbol} @ {price}")
        
        # In a real implementation, this would call the exchange API
        result = self.exchange_api.create_order(
            symbol=symbol,
            side=side,
            order_type="limit",
            quantity=quantity,
            price=price
        )
        
        self.metrics["limit_orders"] += 1
        return result
    
    def _execute_twap_order(self, symbol: str, side: str, quantity: float, price: float, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Time-Weighted Average Price (TWAP) order."""
        self.logger.info(f"TWAP order execution not implemented yet")
        
        # In a real implementation, this would split the order into smaller chunks
        # and execute them over time
        
        self.metrics["twap_orders"] += 1
        return {"error": "TWAP execution not implemented"}
    
    def _execute_iceberg_order(self, symbol: str, side: str, quantity: float, price: float, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an iceberg order (showing only part of the total quantity)."""
        self.logger.info(f"Iceberg order execution not implemented yet")
        
        # In a real implementation, this would execute a series of limit orders
        # showing only a small part of the total quantity at a time
        
        self.metrics["iceberg_orders"] += 1
        return {"error": "Iceberg execution not implemented"}
    
    def _apply_slippage(self, symbol: str, side: str, price: Optional[float]) -> Optional[float]:
        """
        Apply slippage model to price based on market conditions.
        
        Args:
            symbol: Trading pair
            side: Order side (buy/sell)
            price: Original price
            
        Returns:
            Adjusted price with slippage applied
        """
        if not price:
            return None
            
        # Simple slippage model based on configuration
        slippage_factor = self.max_slippage_pct / 100.0
        
        if self.slippage_model == "conservative":
            # Conservative model applies maximum slippage
            if side == "buy":
                return price * (1 + slippage_factor)
            else:
                return price * (1 - slippage_factor)
                
        elif self.slippage_model == "optimistic":
            # Optimistic model applies minimal slippage
            if side == "buy":
                return price * (1 + slippage_factor * 0.25)
            else:
                return price * (1 - slippage_factor * 0.25)
                
        elif self.slippage_model == "realistic":
            # Realistic model varies slippage between 25-75% of max
            import random
            variation = random.uniform(0.25, 0.75)
            if side == "buy":
                return price * (1 + slippage_factor * variation)
            else:
                return price * (1 - slippage_factor * variation)
                
        # Default case
        return price
    
    def _update_metrics(self, executed_orders: List[Dict[str, Any]]) -> None:
        """Update performance metrics based on executed orders."""
        if not executed_orders:
            return
            
        self.metrics["orders_executed"] += len(executed_orders)
        
        # Calculate average slippage
        total_slippage = 0
        slippage_count = 0
        
        for result in executed_orders:
            execution = result.get("execution", {})
            order = result.get("original_order", {})
            
            # Calculate slippage if we have both expected and actual prices
            if "price" in execution and "price" in order and order["price"]:
                expected_price = order["price"]
                actual_price = execution["price"]
                
                if order["side"] == "buy":
                    slippage = (actual_price / expected_price - 1) * 100
                else:
                    slippage = (expected_price / actual_price - 1) * 100
                    
                total_slippage += slippage
                slippage_count += 1
                
            # Track fees
            fee = execution.get("fee", {}).get("cost", 0)
            self.metrics["total_fees"] += fee
            
        # Update average slippage
        if slippage_count > 0:
            self.metrics["avg_slippage"] = total_slippage / slippage_count
    
    def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Check the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Dictionary with order status
        """
        if not self.exchange_api:
            return {"error": "No exchange API available"}
            
        try:
            status = self.exchange_api.get_order_status(order_id)
            
            # If order is no longer active, remove from tracking
            if status.get("status") in ["filled", "canceled", "expired"]:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                    
            return status
            
        except Exception as e:
            self.logger.error(f"Error checking order status: {str(e)}")
            return {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an active order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary with cancellation result
        """
        if not self.exchange_api:
            return {"error": "No exchange API available"}
            
        if order_id not in self.active_orders:
            return {"error": f"Order {order_id} not found in active orders"}
            
        try:
            result = self.exchange_api.cancel_order(order_id)
            
            # If successfully canceled, update tracking
            if result.get("success", False):
                self.active_orders[order_id]["execution"]["status"] = "canceled"
                self.active_orders[order_id]["execution"]["canceled_at"] = datetime.now().isoformat()
                del self.active_orders[order_id]
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            return {"error": str(e)}
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all active orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            Dictionary with cancellation results
        """
        if not self.exchange_api:
            return {"error": "No exchange API available"}
            
        to_cancel = list(self.active_orders.keys())
        if symbol:
            to_cancel = [order_id for order_id, data in self.active_orders.items() 
                        if data["order"].get("symbol") == symbol]
            
        if not to_cancel:
            return {"message": "No active orders to cancel", "canceled": []}
            
        canceled = []
        failed = []
        
        for order_id in to_cancel:
            try:
                result = self.cancel_order(order_id)
                if result.get("success", False):
                    canceled.append(order_id)
                else:
                    failed.append({
                        "order_id": order_id,
                        "error": result.get("error", "Unknown error")
                    })
            except Exception as e:
                failed.append({
                    "order_id": order_id,
                    "error": str(e)
                })
                
        return {
            "canceled": canceled,
            "failed": failed,
            "message": f"Canceled {len(canceled)} orders, {len(failed)} failed"
        }
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history of executed orders.
        
        Args:
            symbol: Optional symbol to filter by
            limit: Maximum number of orders to return
            
        Returns:
            List of order execution records
        """
        history = self.order_history.copy()
        
        if symbol:
            history = [order for order in history 
                      if order.get("order", {}).get("symbol") == symbol]
            
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit and len(history) > limit:
            history = history[:limit]
            
        return history
    
    def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize execution strategies based on historical data.
        
        Args:
            training_data: Historical market data and execution results
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training execution agent with historical data")
        
        # In a real implementation, this could:
        # 1. Analyze historical slippage to tune slippage models
        # 2. Determine optimal execution strategies for different market conditions
        # 3. Adjust retry parameters based on historical success rates
        
        # For now, just simulate an improvement
        old_slippage = self.max_slippage_pct
        
        # Simulate finding a better max slippage parameter
        self.max_slippage_pct = max(0.1, old_slippage * 0.9)  # Reduce by 10%
        
        self.logger.info(f"Updated max slippage: {old_slippage}% -> {self.max_slippage_pct}%")
        
        return {
            "success": True,
            "old_params": {"max_slippage_pct": old_slippage},
            "new_params": {"max_slippage_pct": self.max_slippage_pct},
            "message": "Optimized slippage parameters based on historical data"
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the execution agent."""
        metrics = self.metrics.copy()
        metrics["active_orders"] = len(self.active_orders)
        return metrics
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the execution agent.
        
        Returns:
            Dictionary with state data to be persisted
        """
        state = {
            "metrics": self.metrics,
            "order_history": self.order_history,
            "active_orders": self.active_orders,
            "parameters": {
                "max_slippage_pct": self.max_slippage_pct,
                "slippage_model": self.slippage_model,
                "retry_attempts": self.retry_attempts,
                "retry_delay": self.retry_delay
            }
        }
        
        # Also save wallet state if available
        if self.wallet:
            wallet_path = f"db/wallet_{self.agent_id}.json"
            self.wallet.save_to_file(wallet_path)
            state["wallet_file"] = wallet_path
        
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
            if "metrics" in state:
                self.metrics = state["metrics"]
                
            if "order_history" in state:
                self.order_history = state["order_history"]
                
            if "active_orders" in state:
                self.active_orders = state["active_orders"]
                
            if "parameters" in state:
                params = state["parameters"]
                if "max_slippage_pct" in params:
                    self.max_slippage_pct = params["max_slippage_pct"]
                if "slippage_model" in params:
                    self.slippage_model = params["slippage_model"]
                if "retry_attempts" in params:
                    self.retry_attempts = params["retry_attempts"]
                if "retry_delay" in params:
                    self.retry_delay = params["retry_delay"]
            
            # Load wallet if path is provided
            if "wallet_file" in state:
                try:
                    self.wallet = Wallet.load_from_file(state["wallet_file"])
                except Exception as e:
                    self.logger.error(f"Failed to load wallet: {str(e)}")
                    # Create a new wallet if loading fails
                    self.wallet = Wallet(
                        initial_balance=self.config.get("initial_balance", 10000.0),
                        base_currency=self.config.get("base_currency", "USDT"),
                        name=f"Execution-{self.agent_id}"
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading execution agent state: {str(e)}")
            return False


class DummyExchangeAPI:
    """Simple dummy exchange API for testing purposes."""
    
    def __init__(self, exchange_name, testnet=True):
        """Initialize the dummy exchange API."""
        self.exchange_name = exchange_name
        self.testnet = testnet
        self.order_counter = 1000
        self.logger = logging.getLogger(f"dummy_exchange_{exchange_name}")
        self.logger.info(f"Initialized dummy {exchange_name} API in {'testnet' if testnet else 'live'} mode")
    
    def create_order(self, symbol, side, order_type, quantity, price=None):
        """Create a dummy order."""
        import random
        
        # Generate a unique order ID
        order_id = f"{self.exchange_name}-{self.order_counter}"
        self.order_counter += 1
        
        # For market orders, simulate a price
        if order_type == "market" or not price:
            # Create a random price around 50k for BTC, 3k for ETH, etc.
            base_price = {
                "BTC/USDT": 50000,
                "ETH/USDT": 3000,
                "BNB/USDT": 400,
                "SOL/USDT": 100,
                "ADA/USDT": 1.2,
                "DOGE/USDT": 0.15
            }.get(symbol, 100)
            
            price = base_price * (1 + random.uniform(-0.01, 0.01))
        
        # Calculate value
        value = quantity * price
        
        # Simulate small fee (0.1%)
        fee = value * 0.001
        
        # Create trade result
        result = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "price": price,
            "value": value,
            "fee": {
                "cost": fee,
                "currency": symbol.split('/')[1] if '/' in symbol else "USDT"
            },
            "status": "filled",
            "filled": quantity,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def get_order_status(self, order_id):
        """Get dummy order status."""
        # In a real implementation, this would check the exchange API
        return {"id": order_id, "status": "filled"}
    
    def cancel_order(self, order_id):
        """Cancel a dummy order."""
        # In a real implementation, this would call the exchange API
        return {"success": True, "id": order_id} 