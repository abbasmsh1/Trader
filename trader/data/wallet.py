from typing import Dict, List
from datetime import datetime
import pandas as pd

class Wallet:
    """
    Wallet implementation for the trading system.
    Manages balances, holdings, and trade history for an agent.
    """
    def __init__(self, initial_balance_usdt: float = 20.0):
        """
        Initialize the wallet.
        
        Args:
            initial_balance_usdt: Initial balance in USDT
        """
        self.initial_balance_usdt = initial_balance_usdt
        self.balance_usdt = initial_balance_usdt
        self.holdings: Dict[str, float] = {}  # symbol -> amount
        self.trade_history: List[Dict] = []
        self.performance_history: List[Dict] = []
        self.stop_losses: Dict[str, Dict] = {}  # symbol -> {price: float, percentage: float}
        
    def can_buy(self, symbol: str, amount_usdt: float, current_price: float) -> bool:
        """
        Check if the wallet has enough USDT to buy.
        
        Args:
            symbol: Trading pair symbol
            amount_usdt: Amount in USDT to spend
            current_price: Current price of the asset
            
        Returns:
            True if the wallet has enough USDT, False otherwise
        """
        return amount_usdt <= self.balance_usdt
    
    def can_sell(self, symbol: str, amount_crypto: float) -> bool:
        """
        Check if the wallet has enough crypto to sell.
        
        Args:
            symbol: Trading pair symbol
            amount_crypto: Amount of crypto to sell
            
        Returns:
            True if the wallet has enough crypto, False otherwise
        """
        return symbol in self.holdings and amount_crypto <= self.holdings[symbol]
    
    def execute_buy(self, symbol: str, amount_usdt: float, price: float) -> bool:
        """
        Execute a buy order.
        
        Args:
            symbol: Trading pair symbol
            amount_usdt: Amount in USDT to spend
            price: Price of the asset
            
        Returns:
            True if the buy was successful, False otherwise
        """
        # Validate inputs
        if price <= 0:
            print(f"Error: Invalid price {price} for {symbol}")
            return False
            
        if amount_usdt <= 0:
            print(f"Error: Invalid amount {amount_usdt} for {symbol}")
            return False
            
        if not self.can_buy(symbol, amount_usdt, price):
            return False
            
        crypto_amount = amount_usdt / price
        self.balance_usdt -= amount_usdt
        
        if symbol in self.holdings:
            self.holdings[symbol] += crypto_amount
        else:
            self.holdings[symbol] = crypto_amount
            
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'amount_usdt': amount_usdt,
            'amount_crypto': crypto_amount,
            'price': price
        })
        
        # Set stop loss at 5% below purchase price
        self.set_stop_loss(symbol, price, 0.05)
        
        return True
    
    def execute_sell(self, symbol: str, amount_crypto: float, price: float) -> bool:
        """
        Execute a sell order.
        
        Args:
            symbol: Trading pair symbol
            amount_crypto: Amount of crypto to sell
            price: Price of the asset
            
        Returns:
            True if the sell was successful, False otherwise
        """
        # Validate inputs
        if price <= 0:
            print(f"Error: Invalid price {price} for {symbol}")
            return False
            
        if amount_crypto <= 0:
            print(f"Error: Invalid amount {amount_crypto} for {symbol}")
            return False
            
        if not self.can_sell(symbol, amount_crypto):
            return False
            
        amount_usdt = amount_crypto * price
        self.balance_usdt += amount_usdt
        self.holdings[symbol] -= amount_crypto
        
        # Remove the symbol if the amount is zero
        if self.holdings[symbol] <= 0:
            del self.holdings[symbol]
            
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'amount_usdt': amount_usdt,
            'amount_crypto': amount_crypto,
            'price': price
        })
        
        return True
        
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate the total value of the wallet in USDT.
        
        Args:
            current_prices: Dictionary of current prices (symbol -> price)
            
        Returns:
            Total value in USDT
        """
        total = self.balance_usdt
        
        for symbol, amount in self.holdings.items():
            if symbol in current_prices:
                total += amount * current_prices[symbol]
                
        return total
        
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get performance metrics for the wallet.
        
        Args:
            current_prices: Dictionary of current prices (symbol -> price)
            
        Returns:
            Dictionary of performance metrics
        """
        # Update stop losses
        executed_stop_losses = self.update_stop_losses(current_prices)
        
        # Calculate total value
        total_value = self.get_total_value(current_prices)
        
        # Calculate profit/loss
        profit_loss = total_value - self.initial_balance_usdt
        profit_loss_pct = (profit_loss / self.initial_balance_usdt) * 100 if self.initial_balance_usdt > 0 else 0
        
        # Calculate holdings with prices
        holdings_with_prices = {}
        
        for symbol, amount in self.holdings.items():
            price = current_prices.get(symbol, 0)
            value_usdt = amount * price
            
            holdings_with_prices[symbol] = {
                'amount': amount,
                'price': price,
                'value_usdt': value_usdt
            }
            
            # Add stop loss info if exists
            if symbol in self.stop_losses:
                holdings_with_prices[symbol]['stop_loss'] = self.stop_losses[symbol]['price']
                holdings_with_prices[symbol]['stop_loss_pct'] = self.stop_losses[symbol]['percentage'] * 100
            
        # Get trade statistics
        buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        stop_loss_trades = [t for t in self.trade_history if t['action'] == 'STOP_LOSS']
        
        metrics = {
            'balance_usdt': self.balance_usdt,
            'holdings_value': total_value - self.balance_usdt,
            'total_value': total_value,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'holdings_with_prices': holdings_with_prices,
            'trade_count': len(self.trade_history),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'stop_loss_count': len(stop_loss_trades),
            'active_stop_losses': len(self.stop_losses)
        }
        
        return metrics
        
    def get_position_size(self, symbol: str) -> float:
        """
        Get the position size for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Position size in crypto units
        """
        return self.holdings.get(symbol, 0.0)
        
    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame.
        
        Returns:
            DataFrame of trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
            
        return pd.DataFrame(self.trade_history)
        
    def reset(self):
        """Reset the wallet to its initial state."""
        self.balance_usdt = self.initial_balance_usdt
        self.holdings = {}
        # Keep trade history for reference
        # self.trade_history = []
        # Keep performance history for reference
        # self.performance_history = []
        
    def execute_trade(self, symbol: str, signal: Dict, current_price: float) -> bool:
        """
        Execute a trade based on a signal.
        
        Args:
            symbol: Trading pair symbol
            signal: Signal dictionary with action and amount
            current_price: Current price of the asset
            
        Returns:
            True if the trade was successful, False otherwise
        """
        action = signal.get('action', '').upper()
        
        if action in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
            amount_usdt = signal.get('amount_usdt', 0.0)
            return self.execute_buy(symbol, amount_usdt, current_price)
            
        elif action in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
            amount_crypto = signal.get('amount_crypto', 0.0)
            return self.execute_sell(symbol, amount_crypto, current_price)
            
        return False  # No action or unknown action 

    def set_stop_loss(self, symbol: str, price: float, percentage: float = 0.05) -> None:
        """
        Set a stop loss for a symbol.
        
        Args:
            symbol: Trading pair symbol
            price: Entry price
            percentage: Stop loss percentage (default 5%)
        """
        self.stop_losses[symbol] = {
            'price': price * (1 - percentage),
            'percentage': percentage,
            'entry_price': price,
            'highest_price': price,
            'trailing': True
        }
        
    def update_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Update trailing stop losses and execute if triggered.
        
        Args:
            current_prices: Dictionary of current prices (symbol -> price)
            
        Returns:
            List of executed stop loss trades
        """
        executed_trades = []
        
        for symbol, stop_loss in list(self.stop_losses.items()):
            if symbol not in current_prices or symbol not in self.holdings:
                # Remove stop loss if we don't have the holding anymore
                if symbol in self.stop_losses:
                    del self.stop_losses[symbol]
                continue
                
            current_price = current_prices[symbol]
            
            # Update trailing stop if price has increased
            if stop_loss['trailing'] and current_price > stop_loss['highest_price']:
                # Update highest price
                stop_loss['highest_price'] = current_price
                # Update stop loss price
                stop_loss['price'] = current_price * (1 - stop_loss['percentage'])
                
            # Check if stop loss is triggered
            if current_price <= stop_loss['price']:
                # Execute stop loss
                amount_crypto = self.holdings[symbol]
                success = self.execute_sell(symbol, amount_crypto, current_price)
                
                if success:
                    executed_trades.append({
                        'symbol': symbol,
                        'action': 'STOP_LOSS',
                        'amount': amount_crypto,
                        'price': current_price,
                        'value': amount_crypto * current_price,
                        'entry_price': stop_loss['entry_price'],
                        'highest_price': stop_loss['highest_price'],
                        'stop_price': stop_loss['price']
                    })
                    
                    # Remove stop loss
                    del self.stop_losses[symbol]
                    
        return executed_trades 