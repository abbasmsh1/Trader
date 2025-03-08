import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MockWallet:
    def __init__(self, initial_balance=20.0):
        self.balance_usdt = initial_balance * 0.2  # 20% in USDT
        self.holdings = {
            'BTC/USDT': {'amount': 0.0002, 'price': 65000.0},
            'ETH/USDT': {'amount': 0.01, 'price': 3500.0}
        }
    
    def get_performance_metrics(self, current_prices):
        holdings_with_prices = {}
        total_value_usdt = self.balance_usdt
        
        for symbol, holding in self.holdings.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                value_usdt = holding['amount'] * price
                total_value_usdt += value_usdt
                holdings_with_prices[symbol] = {
                    'amount': holding['amount'],
                    'price': price,
                    'value_usdt': value_usdt
                }
        
        return {
            'total_value_usdt': total_value_usdt,
            'balance_usdt': self.balance_usdt,
            'holdings_with_prices': holdings_with_prices
        }

class MockAgent:
    def __init__(self, name, personality):
        self.name = name
        self._personality = personality
        self.wallet = MockWallet()
    
    def get_personality_traits(self):
        return {'personality': self._personality}

class TradingSystem:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        self.agents = [
            MockAgent("Warren Buffett AI", "Value"),
            MockAgent("Elon Musk AI", "Tech"),
            MockAgent("Technical Trader AI", "Technical")
        ]
        self._mock_prices = {
            'BTC/USDT': 65000.0,
            'ETH/USDT': 3500.0,
            'BNB/USDT': 420.0,
            'SOL/USDT': 110.0
        }
    
    def get_market_data(self, symbol):
        if symbol not in self._mock_prices:
            return pd.DataFrame()
        
        # Create mock price data
        now = datetime.now()
        dates = [now - timedelta(minutes=i) for i in range(60)]
        prices = np.random.normal(self._mock_prices[symbol], self._mock_prices[symbol] * 0.01, 60)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 60)
        }, index=dates)
        
        return df
    
    def _save_state(self):
        """Mock save state method."""
        pass 