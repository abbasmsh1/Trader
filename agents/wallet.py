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
        if not self.can_sell(symbol, amount_crypto):
            return False
            
        amount_usdt = amount_crypto * price
        self.balance_usdt += amount_usdt
        
        # Update holdings
        self.holdings[symbol] -= amount_crypto
        
        # Remove the symbol if all coins are sold or only dust remains
        if self.holdings[symbol] <= 1e-8:  # Increased dust threshold slightly
            del self.holdings[symbol]
            
        self.trades_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'amount_usdt': amount_usdt,
            'amount_crypto': amount_crypto,
            'price': price
        })
        
        return True
    
    def get_total_value_usdt(self, current_prices: Dict[str, float]) -> float:
        """Calculate total wallet value in USDT."""
        total = self.balance_usdt
        
        # Only include non-zero holdings
        for symbol, amount in self.holdings.items():
            if amount > 1e-8 and symbol in current_prices:  # Check for dust
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
        
        # Only include non-dust holdings
        holdings_with_prices = {}
        for symbol, amount in self.holdings.items():
            if amount > 1e-8 and symbol in current_prices:  # Check for dust
                holdings_with_prices[symbol] = {
                    'amount': amount,
                    'price': current_prices[symbol],
                    'value_usdt': amount * current_prices[symbol]
                }
        
        return {
            'total_value_usdt': total_value,
            'balance_usdt': self.balance_usdt,
            'holdings': {k: v for k, v in self.holdings.items() if v > 1e-8},  # Filter out dust
            'holdings_with_prices': holdings_with_prices,
            'total_return_pct': total_return,
            'trade_count': len(self.trades_history),
            'realized_pnl': self.balance_usdt - self.initial_balance_usdt
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as a DataFrame."""
        return pd.DataFrame(self.trades_history) 