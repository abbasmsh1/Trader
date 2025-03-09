from typing import Dict, List
from datetime import datetime
import pandas as pd

class Wallet:
    def __init__(self, initial_balance_usdt: float = 20.0):
        self.initial_balance_usdt = initial_balance_usdt
        self.balance_usdt = initial_balance_usdt
        self.holdings: Dict[str, float] = {}  # symbol -> amount
        self.trades_history: List[Dict] = []
        
    def can_buy(self, symbol: str, amount_usdt: float, current_price: float) -> bool:
        """Check if the wallet has enough USDT to buy."""
        return amount_usdt <= self.balance_usdt
    
    def can_sell(self, symbol: str, amount_crypto: float) -> bool:
        """Check if the wallet has enough crypto to sell."""
        return symbol in self.holdings and amount_crypto <= self.holdings[symbol]
    
    def execute_buy(self, symbol: str, amount_usdt: float, price: float) -> bool:
        """Execute a buy order."""
        if not self.can_buy(symbol, amount_usdt, price):
            return False
            
        crypto_amount = amount_usdt / price
        self.balance_usdt -= amount_usdt
        
        if symbol in self.holdings:
            self.holdings[symbol] += crypto_amount
        else:
            self.holdings[symbol] = crypto_amount
            
        self.trades_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'amount_usdt': amount_usdt,
            'amount_crypto': crypto_amount,
            'price': price
        })
        
        return True
    
    def execute_sell(self, symbol: str, amount_crypto: float, price: float) -> bool:
        """Execute a sell order."""
        # Validate inputs
        if price <= 0:
            print(f"Error: Invalid price {price} for {symbol}")
            return False
            
        if amount_crypto <= 0:
            print(f"Error: Invalid amount {amount_crypto} for {symbol}")
            return False
            
        # Check if we can sell
        if not self.can_sell(symbol, amount_crypto):
            print(f"Cannot sell {amount_crypto} {symbol}: insufficient balance (have {self.holdings.get(symbol, 0)})")
            return False
            
        # Calculate USDT amount
        amount_usdt = amount_crypto * price
        
        # Update balances
        self.balance_usdt += amount_usdt
        self.holdings[symbol] -= amount_crypto
        
        # Remove dust
        if self.holdings[symbol] < 1e-8:
            print(f"Removing dust amount of {symbol}: {self.holdings[symbol]}")
            del self.holdings[symbol]
            
        # Record the trade
        self.trades_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'amount_usdt': amount_usdt,
            'amount_crypto': amount_crypto,
            'price': price
        })
        
        print(f"Sold {amount_crypto:.8f} {symbol} for ${amount_usdt:.2f} at ${price:.2f}")
        return True
    
    def get_total_value_usdt(self, current_prices: Dict[str, float]) -> float:
        """Calculate total wallet value in USDT."""
        total = self.balance_usdt
        for symbol, amount in self.holdings.items():
            if symbol in current_prices:
                total += amount * current_prices[symbol]
        return total
    
    def get_position_size(self, symbol: str) -> float:
        """Get the amount of a specific cryptocurrency held."""
        return self.holdings.get(symbol, 0.0)
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate wallet performance metrics."""
        total_value = self.get_total_value_usdt(current_prices)
        total_return = ((total_value - self.initial_balance_usdt) / 
                       self.initial_balance_usdt * 100)
        
        # Create holdings with prices for display
        holdings_with_prices = {}
        for symbol, amount in self.holdings.items():
            price = current_prices.get(symbol.split('/')[0] if '/' in symbol else symbol, 0)
            holdings_with_prices[symbol] = {
                'amount': amount,
                'price': price,
                'value_usdt': amount * price
            }
        
        return {
            'total_value_usdt': total_value,
            'balance_usdt': self.balance_usdt,
            'holdings': self.holdings.copy(),
            'holdings_with_prices': holdings_with_prices,
            'total_return_pct': total_return,
            'trade_count': len(self.trades_history),
            'realized_pnl': self.balance_usdt - self.initial_balance_usdt
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as a DataFrame."""
        return pd.DataFrame(self.trades_history)
    
    def execute_trade(self, symbol: str, signal: Dict, current_price: float) -> bool:
        """Execute a trade based on the signal."""
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0.5)
        
        # Skip if action is HOLD or WATCH
        if action in ['HOLD', 'WATCH']:
            return False
            
        # Determine trade size based on confidence and action
        if action in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
            # Calculate amount to buy based on confidence
            if action == 'STRONG_BUY':
                amount_usdt = self.balance_usdt * 0.5 * confidence  # Use up to 50% of balance for strong buy
            elif action == 'BUY':
                amount_usdt = self.balance_usdt * 0.3 * confidence  # Use up to 30% of balance for buy
            else:  # SCALE_IN
                amount_usdt = self.balance_usdt * 0.2 * confidence  # Use up to 20% of balance for scaling in
                
            # Ensure minimum trade size
            amount_usdt = max(amount_usdt, 1.0) if self.balance_usdt >= 5.0 else self.balance_usdt * 0.2
            
            # Execute buy
            return self.execute_buy(symbol, amount_usdt, current_price)
            
        elif action in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
            # Handle different symbol formats (with or without /USDT)
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Check if we have the asset in either format
            holding_symbol = None
            if base_symbol in self.holdings and self.holdings[base_symbol] > 0:
                holding_symbol = base_symbol
            elif symbol in self.holdings and self.holdings[symbol] > 0:
                holding_symbol = symbol
            
            # If we don't have the asset, return False
            if not holding_symbol:
                return False
                
            # Calculate amount to sell based on confidence
            if action == 'STRONG_SELL':
                amount_crypto = self.holdings[holding_symbol] * 0.8 * confidence  # Sell up to 80% of holdings for strong sell
            elif action == 'SELL':
                amount_crypto = self.holdings[holding_symbol] * 0.5 * confidence  # Sell up to 50% of holdings for sell
            else:  # SCALE_OUT
                amount_crypto = self.holdings[holding_symbol] * 0.3 * confidence  # Sell up to 30% of holdings for scaling out
                
            # Ensure minimum trade size and don't exceed holdings
            amount_crypto = min(amount_crypto, self.holdings[holding_symbol])
            
            # Check if the amount is too small (dust)
            if amount_crypto <= 1e-8:
                return False
                
            # Execute sell with the correct symbol
            return self.execute_sell(holding_symbol, amount_crypto, current_price)
            
        return False 