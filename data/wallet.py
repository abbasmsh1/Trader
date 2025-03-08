from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid

class Wallet:
    def __init__(self, initial_balance_usdt: float = 20.0, weekly_profit_goal_pct: float = 10.0):
        self.wallet_id = str(uuid.uuid4())[:8]  # Unique wallet ID
        self.initial_balance_usdt = initial_balance_usdt
        self.balance_usdt = initial_balance_usdt
        self.holdings: Dict[str, float] = {}  # symbol -> amount
        self.trades_history: List[Dict] = []
        self.value_history: List[Dict] = []  # Track wallet value over time
        self.creation_date = datetime.now()
        self.weekly_profit_goal_pct = weekly_profit_goal_pct
        self.weekly_profit_goal_usdt = initial_balance_usdt * (weekly_profit_goal_pct / 100)
        
        # Record initial value
        self.value_history.append({
            'timestamp': self.creation_date,
            'total_value_usdt': initial_balance_usdt,
            'balance_usdt': initial_balance_usdt,
            'holdings_value': 0.0,
            'holdings_count': 0
        })
        
    def get_balance(self, asset: str) -> float:
        """Get balance of a specific asset."""
        if asset == 'USDT':
            return self.balance_usdt
        return self.holdings.get(asset, 0.0)
        
    def can_buy(self, symbol: str, amount_crypto: float, current_price: float) -> bool:
        """Check if the wallet has enough USDT to buy."""
        amount_usdt = amount_crypto * current_price
        return amount_usdt <= self.balance_usdt and amount_usdt > 0
    
    def can_sell(self, symbol: str, amount_crypto: float) -> bool:
        """Check if the wallet has enough crypto to sell."""
        return symbol in self.holdings and amount_crypto <= self.holdings[symbol] and amount_crypto > 0
    
    def execute_buy(self, symbol: str, amount_crypto: float, price: float) -> bool:
        """
        Execute a buy order.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            amount_crypto (float): Amount of cryptocurrency to buy
            price (float): Current price of the cryptocurrency
            
        Returns:
            bool: Whether the buy was successful
        """
        amount_usdt = amount_crypto * price
        
        if not self.can_buy(symbol, amount_crypto, price):
            return False
            
        self.balance_usdt -= amount_usdt
        
        # Extract base asset from symbol (e.g., 'BTC' from 'BTC/USDT')
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol
        
        if base_asset in self.holdings:
            self.holdings[base_asset] += amount_crypto
        else:
            self.holdings[base_asset] = amount_crypto
            
        # Record the trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'amount_usdt': amount_usdt,
            'amount_crypto': amount_crypto,
            'price': price,
            'fee': amount_usdt * 0.001  # Assume 0.1% fee
        }
        self.trades_history.append(trade)
        
        # Update wallet value history
        self._update_value_history({symbol.split('/')[0]: price})
        
        return True
    
    def execute_sell(self, symbol: str, amount_crypto: float, price: float) -> bool:
        """
        Execute a sell order.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            amount_crypto (float): Amount of cryptocurrency to sell
            price (float): Current price of the cryptocurrency
            
        Returns:
            bool: Whether the sell was successful
        """
        # Extract base asset from symbol (e.g., 'BTC' from 'BTC/USDT')
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol
        
        if not self.can_sell(base_asset, amount_crypto):
            return False
            
        amount_usdt = amount_crypto * price
        fee = amount_usdt * 0.001  # Assume 0.1% fee
        net_amount = amount_usdt - fee
        
        self.balance_usdt += net_amount
        self.holdings[base_asset] -= amount_crypto
        
        if self.holdings[base_asset] < 1e-8:  # Remove dust
            del self.holdings[base_asset]
            
        # Calculate profit/loss for this trade
        avg_buy_price = self._calculate_avg_buy_price(base_asset)
        profit_loss = (price - avg_buy_price) * amount_crypto if avg_buy_price > 0 else 0
            
        # Record the trade
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'amount_usdt': amount_usdt,
            'amount_crypto': amount_crypto,
            'price': price,
            'fee': fee,
            'profit_loss': profit_loss
        }
        self.trades_history.append(trade)
        
        # Update wallet value history
        self._update_value_history({symbol.split('/')[0]: price})
        
        return True
    
    def _calculate_avg_buy_price(self, asset: str) -> float:
        """Calculate average buy price for an asset."""
        buy_trades = [t for t in self.trades_history 
                     if t['action'] == 'BUY' and t['symbol'].startswith(asset)]
        
        if not buy_trades:
            return 0.0
            
        total_amount = sum(t['amount_crypto'] for t in buy_trades)
        total_cost = sum(t['amount_usdt'] for t in buy_trades)
        
        return total_cost / total_amount if total_amount > 0 else 0.0
    
    def _update_value_history(self, current_prices: Dict[str, float]):
        """Update wallet value history with current state."""
        holdings_value = sum(
            amount * current_prices.get(symbol, 0) 
            for symbol, amount in self.holdings.items()
        )
        
        self.value_history.append({
            'timestamp': datetime.now(),
            'total_value_usdt': self.balance_usdt + holdings_value,
            'balance_usdt': self.balance_usdt,
            'holdings_value': holdings_value,
            'holdings_count': len(self.holdings)
        })
    
    def get_total_value_usdt(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total wallet value in USDT.
        
        Args:
            current_prices (Dict[str, float]): Current prices of assets
            
        Returns:
            float: Total wallet value in USDT
        """
        total = self.balance_usdt
        for symbol, amount in self.holdings.items():
            if symbol in current_prices:
                total += amount * current_prices[symbol]
        return total
    
    def get_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get comprehensive wallet metrics.
        
        Args:
            current_prices (Dict[str, float]): Current prices of assets
            
        Returns:
            Dict: Wallet metrics including performance, holdings, and goals
        """
        # Calculate current total value
        total_value = self.get_total_value_usdt(current_prices)
        
        # Calculate returns
        total_return = ((total_value - self.initial_balance_usdt) / 
                       self.initial_balance_usdt * 100)
        
        # Calculate time-based metrics
        days_active = (datetime.now() - self.creation_date).days
        
        # Calculate weekly performance
        weekly_return = self._calculate_weekly_return()
        
        # Calculate goal progress
        weekly_goal_progress = (weekly_return / self.weekly_profit_goal_pct) * 100
        
        # Calculate holdings breakdown
        holdings_breakdown = []
        for symbol, amount in self.holdings.items():
            if symbol in current_prices:
                value = amount * current_prices[symbol]
                holdings_breakdown.append({
                    'symbol': symbol,
                    'amount': amount,
                    'value_usdt': value,
                    'percentage': (value / total_value) * 100 if total_value > 0 else 0
                })
        
        # Calculate trade statistics
        buy_trades = [t for t in self.trades_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades_history if t['action'] == 'SELL']
        
        # Calculate realized profit/loss
        realized_pnl = sum(t.get('profit_loss', 0) for t in sell_trades)
        
        return {
            'wallet_id': self.wallet_id,
            'total_value_usdt': total_value,
            'balance_usdt': self.balance_usdt,
            'holdings': self.holdings.copy(),
            'holdings_breakdown': holdings_breakdown,
            'total_return_pct': total_return,
            'days_active': days_active,
            'weekly_return_pct': weekly_return,
            'weekly_goal_pct': self.weekly_profit_goal_pct,
            'weekly_goal_progress_pct': weekly_goal_progress,
            'trade_count': len(self.trades_history),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': total_value - self.initial_balance_usdt - realized_pnl
        }
    
    def _calculate_weekly_return(self) -> float:
        """Calculate return for the current week."""
        # Get start of current week
        now = datetime.now()
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Find value at start of week
        values_before_week = [v for v in self.value_history 
                             if v['timestamp'] < start_of_week]
        
        if not values_before_week:
            # If no data before this week, use initial balance
            start_value = self.initial_balance_usdt
        else:
            # Use the last value before the week started
            start_value = values_before_week[-1]['total_value_usdt']
        
        # Get current value
        current_value = self.value_history[-1]['total_value_usdt'] if self.value_history else self.initial_balance_usdt
        
        # Calculate weekly return
        return ((current_value - start_value) / start_value) * 100 if start_value > 0 else 0
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as a DataFrame."""
        if not self.trades_history:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'action', 'amount_usdt', 
                                        'amount_crypto', 'price', 'fee', 'profit_loss'])
        return pd.DataFrame(self.trades_history)
    
    def get_value_history_df(self) -> pd.DataFrame:
        """Get wallet value history as a DataFrame."""
        if not self.value_history:
            return pd.DataFrame(columns=['timestamp', 'total_value_usdt', 'balance_usdt', 
                                        'holdings_value', 'holdings_count'])
        return pd.DataFrame(self.value_history)
    
    def get_weekly_performance(self) -> Dict:
        """Get detailed weekly performance metrics."""
        df = self.get_value_history_df()
        if df.empty:
            return {
                'weekly_return_pct': 0.0,
                'weekly_goal_pct': self.weekly_profit_goal_pct,
                'weekly_goal_progress_pct': 0.0,
                'days_left_in_week': 7 - datetime.now().weekday(),
                'projected_end_value': self.initial_balance_usdt
            }
        
        # Calculate days left in the week
        days_left_in_week = 7 - datetime.now().weekday()
        
        # Get weekly return
        weekly_return = self._calculate_weekly_return()
        
        # Calculate goal progress
        weekly_goal_progress = (weekly_return / self.weekly_profit_goal_pct) * 100
        
        # Project end of week value based on current trajectory
        current_value = df['total_value_usdt'].iloc[-1]
        daily_growth_rate = weekly_return / (7 - days_left_in_week) if 7 - days_left_in_week > 0 else 0
        projected_end_value = current_value * (1 + (daily_growth_rate / 100) * days_left_in_week)
        
        return {
            'weekly_return_pct': weekly_return,
            'weekly_goal_pct': self.weekly_profit_goal_pct,
            'weekly_goal_progress_pct': weekly_goal_progress,
            'days_left_in_week': days_left_in_week,
            'projected_end_value': projected_end_value
        } 