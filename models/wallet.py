"""
Wallet - Manages cryptocurrency and fiat holdings for the trading system.

This module provides a Wallet class that tracks balances across multiple currencies,
maintains trade history, and calculates portfolio value in real-time.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid

class Wallet:
    """
    Manages cryptocurrency and fiat holdings for the trading system.
    
    The Wallet class provides functionality to:
    - Track balances across multiple currencies
    - Record and query trade history
    - Calculate total portfolio value in real-time
    - Generate performance reports
    """
    
    def __init__(self, 
                 initial_balance: float = 20.0, 
                 base_currency: str = "USDT",
                 name: str = "Main Trading Wallet"):
        """
        Initialize a new wallet.
        
        Args:
            initial_balance: Initial balance in base currency (default: 20.0 USDT)
            base_currency: Base currency symbol (default: USDT)
            name: Wallet name for identification
        """
        self.name = name
        self.base_currency = base_currency
        self.initial_balance = initial_balance  # Store initial balance as instance attribute
        self.creation_time = datetime.now()
        self.last_modified = self.creation_time
        self.wallet_id = str(uuid.uuid4())
        
        # Holdings structure
        self.balances = {
            base_currency: initial_balance
        }
        
        # Trade history
        self.trade_history = []
        
        # Performance tracking
        self.performance_history = []
        self.deposits_withdrawals = []
        
        # Last known prices for value calculation
        self.last_prices = {}
        
        self.logger = logging.getLogger(f"wallet_{self.wallet_id[:8]}")
        self.logger.info(f"Initialized wallet with {initial_balance} {base_currency}")
    
    def get_total_value(self, price_data: Optional[Dict[str, float]] = None) -> float:
        """
        Get the total value of the wallet in the base currency.
        
        This is a simplified version of calculate_total_value that just returns
        the total value as a float for easy display.
        
        Args:
            price_data: Optional dictionary mapping currency symbols to prices in base currency.
                       If not provided, uses last known prices.
        
        Returns:
            Float representing total wallet value in base currency
        """
        # For demo or when no price data is available, just return base currency balance
        if not price_data and not self.last_prices and len(self.balances) == 1:
            return self.balances.get(self.base_currency, 0.0)
            
        # Use provided price data or fallback to last known prices
        prices = price_data or self.last_prices
        
        total_value = 0.0
        
        # Add base currency value directly
        total_value += self.balances.get(self.base_currency, 0.0)
        
        # Add value of other currencies converted to base currency
        for currency, amount in self.balances.items():
            if currency == self.base_currency:
                continue  # Already counted
                
            if currency in prices:
                total_value += amount * prices[currency]
            else:
                # If price not available, we can't include in the calculation
                self.logger.warning(f"No price data for {currency}, excluding from total value calculation")
        
        return total_value
    
    def get_balance(self, currency: str) -> float:
        """
        Get the balance of a specific currency.
        
        Args:
            currency: Currency symbol (e.g., "BTC", "ETH", "USDT")
            
        Returns:
            Float balance of the currency (0.0 if not held)
        """
        return self.balances.get(currency, 0.0)
    
    def get_all_balances(self) -> Dict[str, float]:
        """
        Get all currency balances.
        
        Returns:
            Dictionary mapping currency symbols to balances
        """
        return self.balances.copy()
    
    def update_balance(self, currency: str, amount: float) -> bool:
        """
        Update balance of a specific currency.
        
        Args:
            currency: Currency symbol
            amount: Amount to add (positive) or subtract (negative)
            
        Returns:
            Boolean indicating success
            
        Raises:
            ValueError: If resulting balance would be negative
        """
        current_balance = self.get_balance(currency)
        new_balance = current_balance + amount
        
        if new_balance < 0:
            self.logger.error(f"Cannot update {currency} balance: insufficient funds")
            raise ValueError(f"Insufficient funds: {current_balance} {currency} available, tried to subtract {abs(amount)}")
        
        self.balances[currency] = new_balance
        self.last_modified = datetime.now()
        self.logger.debug(f"Updated {currency} balance: {current_balance} -> {new_balance}")
        
        return True
    
    def add_trade(self, 
                  trade_type: str, 
                  from_currency: str, 
                  to_currency: str,
                  from_amount: float,
                  to_amount: float,
                  price: float,
                  fee: float = 0.0,
                  fee_currency: str = None,
                  exchange: str = "unknown",
                  external_id: str = None) -> Dict[str, Any]:
        """
        Record a trade and update balances accordingly.
        
        Args:
            trade_type: Type of trade ("buy", "sell", "convert")
            from_currency: Currency being spent
            to_currency: Currency being received
            from_amount: Amount of from_currency
            to_amount: Amount of to_currency
            price: Price per unit
            fee: Trading fee
            fee_currency: Currency of fee (defaults to from_currency)
            exchange: Exchange where trade was executed
            external_id: Exchange-provided trade ID
            
        Returns:
            Dictionary with trade details including internal ID
            
        Raises:
            ValueError: If insufficient funds for the trade
        """
        if fee_currency is None:
            fee_currency = from_currency
            
        # Verify sufficient balance for both the trade and fee
        if self.get_balance(from_currency) < from_amount:
            raise ValueError(f"Insufficient {from_currency} for trade")
            
        if fee > 0 and fee_currency == from_currency:
            if self.get_balance(from_currency) < from_amount + fee:
                raise ValueError(f"Insufficient {from_currency} for trade + fee")
        elif fee > 0 and self.get_balance(fee_currency) < fee:
            raise ValueError(f"Insufficient {fee_currency} for fee")
        
        # Generate trade record
        trade_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        trade = {
            "id": trade_id,
            "timestamp": timestamp.isoformat(),
            "type": trade_type,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "from_amount": from_amount,
            "to_amount": to_amount,
            "price": price,
            "fee": fee,
            "fee_currency": fee_currency,
            "exchange": exchange,
            "external_id": external_id
        }
        
        # Update balances
        self.update_balance(from_currency, -from_amount)
        self.update_balance(to_currency, to_amount)
        
        # Handle fee
        if fee > 0:
            self.update_balance(fee_currency, -fee)
        
        # Add to history
        self.trade_history.append(trade)
        self.last_modified = timestamp
        
        self.logger.info(f"Recorded trade: {trade_type} {from_amount} {from_currency} -> {to_amount} {to_currency}")
        
        # Update last known price
        if trade_type == "buy":
            self.last_prices[to_currency] = price
        elif trade_type == "sell":
            self.last_prices[from_currency] = price
        
        return trade
    
    def get_trade_history(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          trade_type: Optional[str] = None,
                          currency: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get filtered trade history.
        
        Args:
            start_time: Filter trades after this time
            end_time: Filter trades before this time
            trade_type: Filter by trade type
            currency: Filter trades involving this currency
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        # Start with full history
        filtered_history = self.trade_history.copy()
        
        # Apply filters
        if start_time:
            filtered_history = [
                trade for trade in filtered_history 
                if datetime.fromisoformat(trade["timestamp"]) >= start_time
            ]
            
        if end_time:
            filtered_history = [
                trade for trade in filtered_history 
                if datetime.fromisoformat(trade["timestamp"]) <= end_time
            ]
            
        if trade_type:
            filtered_history = [
                trade for trade in filtered_history 
                if trade["type"] == trade_type
            ]
            
        if currency:
            filtered_history = [
                trade for trade in filtered_history 
                if trade["from_currency"] == currency or trade["to_currency"] == currency
            ]
        
        # Sort by timestamp (newest first)
        filtered_history.sort(
            key=lambda x: datetime.fromisoformat(x["timestamp"]), 
            reverse=True
        )
        
        # Apply limit
        return filtered_history[:limit]
    
    def update_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update last known prices for value calculation.
        
        Args:
            price_data: Dictionary mapping currency symbols to prices in base currency
        """
        self.last_prices.update(price_data)
        self.logger.debug(f"Updated prices for {len(price_data)} currencies")
    
    def calculate_total_value(self, price_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate total portfolio value in base currency.
        
        Args:
            price_data: Optional current price data (if not provided, uses last known prices)
            
        Returns:
            Dictionary with valuation details
        """
        if price_data:
            self.update_prices(price_data)
            
        total_value = self.get_balance(self.base_currency)
        valuation = {
            "timestamp": datetime.now().isoformat(),
            "base_currency": self.base_currency,
            "base_balance": total_value,
            "holdings": {}
        }
        
        # Calculate value of non-base currencies
        for currency, balance in self.balances.items():
            if currency == self.base_currency or balance == 0:
                continue
                
            price = self.last_prices.get(currency)
            if price is None:
                self.logger.warning(f"No price available for {currency}, skipping in valuation")
                valuation["holdings"][currency] = {
                    "balance": balance,
                    "price": None,
                    "value": None
                }
                continue
                
            value = balance * price
            total_value += value
            
            valuation["holdings"][currency] = {
                "balance": balance,
                "price": price,
                "value": value
            }
        
        valuation["total_value"] = total_value
        
        # Record this valuation in performance history
        self.performance_history.append({
            "timestamp": valuation["timestamp"],
            "total_value": total_value
        })
        
        # Trim performance history if it's getting too long
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
        return valuation
    
    def calculate_performance(self, 
                             period_start: Optional[datetime] = None,
                             period_end: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics for the wallet.
        
        Args:
            period_start: Start of period for calculation (default: wallet creation)
            period_end: End of period for calculation (default: now)
            
        Returns:
            Dictionary with performance metrics
        """
        if period_start is None:
            period_start = self.creation_time
            
        if period_end is None:
            period_end = datetime.now()
            
        # Get history points within the period
        history_points = [
            point for point in self.performance_history
            if datetime.fromisoformat(point["timestamp"]) >= period_start
            and datetime.fromisoformat(point["timestamp"]) <= period_end
        ]
        
        # If no history points in period, return empty metrics
        if not history_points:
            return {
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "initial_value": self.initial_balance,
                "final_value": self.get_total_value(),
                "absolute_return": 0.0,
                "percentage_return": 0.0,
                "annualized_return": 0.0
            }
        
        # Get initial and final values
        history_points.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))
        initial_value = history_points[0]["total_value"]
        final_value = history_points[-1]["total_value"]
        
        # Calculate returns
        absolute_return = final_value - initial_value
        percentage_return = (absolute_return / initial_value) * 100 if initial_value > 0 else 0.0
        
        # Calculate time period in years
        period_start_time = datetime.fromisoformat(history_points[0]["timestamp"])
        period_end_time = datetime.fromisoformat(history_points[-1]["timestamp"])
        years = (period_end_time - period_start_time).total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Calculate annualized return
        if years > 0:
            annualized_return = ((1 + percentage_return / 100) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0
        
        return {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "initial_value": initial_value,
            "final_value": final_value,
            "absolute_return": absolute_return,
            "percentage_return": percentage_return,
            "annualized_return": annualized_return
        }
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save wallet state to a JSON file.
        
        Args:
            filepath: Path to save the wallet state
            
        Returns:
            Boolean indicating success
        """
        try:
            wallet_data = {
                "wallet_id": self.wallet_id,
                "name": self.name,
                "base_currency": self.base_currency,
                "creation_time": self.creation_time.isoformat(),
                "last_modified": self.last_modified.isoformat(),
                "balances": self.balances,
                "trade_history": self.trade_history,
                "deposits_withdrawals": self.deposits_withdrawals,
                "performance_history": self.performance_history,
                "last_prices": self.last_prices
            }
            
            with open(filepath, 'w') as f:
                json.dump(wallet_data, f, indent=2)
                
            self.logger.info(f"Saved wallet state to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving wallet to file: {str(e)}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['Wallet']:
        """
        Load wallet state from a JSON file.
        
        Args:
            filepath: Path to load the wallet state from
            
        Returns:
            Wallet instance or None if loading failed
        """
        try:
            with open(filepath, 'r') as f:
                wallet_data = json.load(f)
            
            # Create a new wallet instance
            wallet = cls(
                initial_balance=0.0,  # Will be overwritten by loaded balances
                base_currency=wallet_data["base_currency"],
                name=wallet_data["name"]
            )
            
            # Restore wallet ID and timestamps
            wallet.wallet_id = wallet_data["wallet_id"]
            wallet.creation_time = datetime.fromisoformat(wallet_data["creation_time"])
            wallet.last_modified = datetime.fromisoformat(wallet_data["last_modified"])
            
            # Restore balances, trade history, etc.
            wallet.balances = wallet_data["balances"]
            wallet.trade_history = wallet_data["trade_history"]
            wallet.deposits_withdrawals = wallet_data["deposits_withdrawals"]
            wallet.performance_history = wallet_data["performance_history"]
            wallet.last_prices = wallet_data["last_prices"]
            
            wallet.logger.info(f"Loaded wallet from {filepath}")
            return wallet
            
        except Exception as e:
            logging.error(f"Error loading wallet from file: {str(e)}")
            return None 