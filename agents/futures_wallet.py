from typing import Dict, List
from datetime import datetime
import pandas as pd
from .wallet import Wallet

class FuturesWallet(Wallet):
    def __init__(self, initial_balance_usdt: float = 20.0, leverage: float = 3.0):
        super().__init__(initial_balance_usdt)
        self.leverage = leverage
        self.positions: Dict[str, Dict] = {}  # symbol -> {size, entry_price, liquidation_price}
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0
        
    def calculate_liquidation_price(self, symbol: str, entry_price: float, position_size: float, is_long: bool) -> float:
        """Calculate the liquidation price for a leveraged position."""
        margin = abs(position_size * entry_price) / self.leverage
        maintenance_margin = margin * 0.5  # 50% maintenance margin requirement
        
        if is_long:
            liquidation_price = entry_price * (1 - (1 / self.leverage) + 0.05)  # 5% buffer
        else:
            liquidation_price = entry_price * (1 + (1 / self.leverage) - 0.05)  # 5% buffer
            
        return liquidation_price
        
    def can_open_position(self, symbol: str, amount_usdt: float, current_price: float) -> bool:
        """Check if we can open a new position with the given margin."""
        required_margin = amount_usdt / self.leverage
        available_margin = self.balance_usdt - self.margin_used
        return required_margin <= available_margin
        
    def open_long(self, symbol: str, amount_usdt: float, price: float) -> bool:
        """Open a long position with leverage."""
        if not self.can_open_position(symbol, amount_usdt, price):
            return False
            
        position_size = (amount_usdt * self.leverage) / price
        margin = amount_usdt / self.leverage
        
        self.margin_used += margin
        self.balance_usdt -= margin
        
        liquidation_price = self.calculate_liquidation_price(symbol, price, position_size, True)
        
        self.positions[symbol] = {
            'size': position_size,
            'entry_price': price,
            'liquidation_price': liquidation_price,
            'is_long': True,
            'margin': margin
        }
        
        self.trades_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'LONG',
            'amount_usdt': amount_usdt,
            'amount_crypto': position_size,
            'price': price,
            'leverage': self.leverage
        })
        
        return True
        
    def open_short(self, symbol: str, amount_usdt: float, price: float) -> bool:
        """Open a short position with leverage."""
        if not self.can_open_position(symbol, amount_usdt, price):
            return False
            
        position_size = (amount_usdt * self.leverage) / price
        margin = amount_usdt / self.leverage
        
        self.margin_used += margin
        self.balance_usdt -= margin
        
        liquidation_price = self.calculate_liquidation_price(symbol, price, position_size, False)
        
        self.positions[symbol] = {
            'size': -position_size,  # Negative for short positions
            'entry_price': price,
            'liquidation_price': liquidation_price,
            'is_long': False,
            'margin': margin
        }
        
        self.trades_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'SHORT',
            'amount_usdt': amount_usdt,
            'amount_crypto': position_size,
            'price': price,
            'leverage': self.leverage
        })
        
        return True
        
    def close_position(self, symbol: str, price: float) -> bool:
        """Close an existing position."""
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        position_size = abs(position['size'])
        is_long = position['is_long']
        margin = position['margin']
        
        # Calculate PNL
        if is_long:
            pnl = position_size * (price - entry_price)
        else:
            pnl = position_size * (entry_price - price)
            
        # Return margin and add PNL to balance
        self.margin_used -= margin
        self.balance_usdt += margin + pnl
        
        self.trades_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'CLOSE',
            'amount_usdt': position_size * price,
            'amount_crypto': position_size,
            'price': price,
            'pnl': pnl,
            'leverage': self.leverage
        })
        
        del self.positions[symbol]
        return True
        
    def update_positions(self, current_prices: Dict[str, float]):
        """Update unrealized PNL and check for liquidations."""
        self.unrealized_pnl = 0.0
        positions_to_liquidate = []
        
        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            position_size = abs(position['size'])
            
            # Check for liquidation
            if position['is_long'] and current_price <= position['liquidation_price']:
                positions_to_liquidate.append((symbol, current_price))
                continue
            elif not position['is_long'] and current_price >= position['liquidation_price']:
                positions_to_liquidate.append((symbol, current_price))
                continue
                
            # Calculate unrealized PNL
            if position['is_long']:
                pnl = position_size * (current_price - position['entry_price'])
            else:
                pnl = position_size * (position['entry_price'] - current_price)
                
            self.unrealized_pnl += pnl
            
        # Handle liquidations
        for symbol, price in positions_to_liquidate:
            self.close_position(symbol, price)
            
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate wallet performance metrics including futures positions."""
        self.update_positions(current_prices)
        
        metrics = super().get_performance_metrics(current_prices)
        total_value = metrics['total_value_usdt'] + self.unrealized_pnl
        
        # Add futures-specific metrics
        metrics.update({
            'total_value_usdt': total_value,
            'unrealized_pnl': self.unrealized_pnl,
            'margin_used': self.margin_used,
            'available_margin': self.balance_usdt - self.margin_used,
            'positions': self.positions,
            'leverage': self.leverage
        })
        
        return metrics 