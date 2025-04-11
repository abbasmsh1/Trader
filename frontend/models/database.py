"""
Database handler for frontend application.
Uses real Binance data and manages trader portfolios.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime
from frontend.services.binance_service import BinanceService

class PickleDBHandler:
    """Handler for PickleDB operations."""
    
    def __init__(self, db_path: str = "data"):
        """Initialize the PickleDB handler."""
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.db_file = os.path.join(db_path, "trading_system.pkl")
        self.load_state()
    
    def load_state(self) -> None:
        """Load the system state from PickleDB."""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    self.state = pickle.load(f)
            else:
                self.state = {
                    'traders': [],
                    'portfolios': {},
                    'trades': [],
                    'market_data': {},
                    'signals': []
                }
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            self.state = {
                'traders': [],
                'portfolios': {},
                'trades': [],
                'market_data': {},
                'signals': []
            }
    
    def save_state(self) -> None:
        """Save the system state to PickleDB."""
        try:
            with open(self.db_file, 'wb') as f:
                pickle.dump(self.state, f)
        except Exception as e:
            print(f"Error saving state: {str(e)}")
    
    def get_traders(self) -> List[Dict[str, Any]]:
        """Get all traders."""
        return self.state.get('traders', [])
    
    def get_portfolio(self, trader_id: str) -> Dict[str, Any]:
        """Get portfolio for a trader."""
        return self.state.get('portfolios', {}).get(trader_id, {})
    
    def get_trades(self, trader_id: str) -> List[Dict[str, Any]]:
        """Get trades for a trader."""
        return [trade for trade in self.state.get('trades', []) if trade.get('trader_id') == trader_id]
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get market data."""
        return self.state.get('market_data', {})

class DatabaseHandler:
    """Database handler for frontend application."""
    
    def __init__(self, db_path: str = "data"):
        """Initialize the database handler with real data."""
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize Binance service
        self.binance = BinanceService()
        
        # Initialize PickleDB handler
        self.pickle_db = PickleDBHandler(db_path)
        
        # Define supported trading pairs
        self.trading_pairs = [
            'BTC/USDT',   # Bitcoin
            'ETH/USDT',   # Ethereum
            'BNB/USDT',   # Binance Coin
            'SOL/USDT',   # Solana
            'ADA/USDT',   # Cardano
            'XRP/USDT',   # Ripple
            'DOT/USDT',   # Polkadot
            'DOGE/USDT',  # Dogecoin
            'AVAX/USDT',  # Avalanche
            'MATIC/USDT', # Polygon
            'LINK/USDT',  # Chainlink
            'UNI/USDT',   # Uniswap
            'ATOM/USDT',  # Cosmos
            'ALGO/USDT',  # Algorand
            'FIL/USDT',   # Filecoin
            'NEAR/USDT',  # NEAR Protocol
            'APE/USDT',   # ApeCoin
            'LTC/USDT',   # Litecoin
            'BCH/USDT',   # Bitcoin Cash
            'AAVE/USDT'   # Aave
        ]
        
        # Initialize traders with $20 each
        self.traders = [
            {'id': 'momentum', 'name': 'Momentum Trader', 'style': 'Trend Following', 'description': 'Follows market trends using momentum indicators'},
            {'id': 'mean_reversion', 'name': 'Mean Reversion Trader', 'style': 'Counter-Trend', 'description': 'Trades price reversions to the mean'},
            {'id': 'breakout', 'name': 'Breakout Trader', 'style': 'Momentum', 'description': 'Captures price breakouts from ranges'},
            {'id': 'grid', 'name': 'Grid Trader', 'style': 'Market Making', 'description': 'Places buy and sell orders in a price grid'},
            {'id': 'swing', 'name': 'Swing Trader', 'style': 'Multi-timeframe', 'description': 'Trades medium-term price swings'},
            {'id': 'scalper', 'name': 'Scalper', 'style': 'High Frequency', 'description': 'Makes quick trades for small profits'},
            {'id': 'arbitrage', 'name': 'Arbitrage Trader', 'style': 'Market Neutral', 'description': 'Profits from price differences across markets'},
            {'id': 'ml_prediction', 'name': 'ML Predictor', 'style': 'Machine Learning', 'description': 'Uses ML models for price prediction'},
            {'id': 'sentiment', 'name': 'Sentiment Trader', 'style': 'News Based', 'description': 'Trades based on market sentiment analysis'},
            {'id': 'options', 'name': 'Options Trader', 'style': 'Derivatives', 'description': 'Trades crypto options and derivatives'}
        ]
        
        # Get current BTC price
        btc_price_data = self.binance.get_current_price('BTC/USDT')
        if not btc_price_data:
            raise Exception("Failed to get BTC price for initialization")
        
        btc_price = btc_price_data['price']
        
        # Initialize portfolios with $20 USDT each and enforce initial BTC purchase
        self.portfolios = {}
        for trader in self.traders:
            # Calculate BTC amount to buy (50% of initial capital)
            btc_amount = 10.00 / btc_price  # $10 worth of BTC
            
            # Create portfolio with initial BTC holding
            self.portfolios[trader['id']] = {
                'total_value': 20.00,
                'base_currency': 'USDT',
                'base_amount': 10.00,  # $10 remaining in USDT
                'assets_value': 10.00,  # $10 worth of BTC
                'num_assets': 1,
                'holdings': [{
                    'symbol': 'BTC/USDT',
                    'amount': btc_amount,
                    'value': 10.00,
                    'price': btc_price,
                    'timestamp': datetime.now().isoformat()
                }]
            }
            
            # Record the initial BTC purchase
            trade = {
                'trader_id': trader['id'],
                'action': 'buy',
                'symbol': 'BTC/USDT',
                'amount': btc_amount,
                'price': btc_price,
                'value': 10.00,
                'timestamp': datetime.now().isoformat()
            }
            self.pickle_db.state['trades'].append(trade)
        
        # Save initial state
        self._save_state()
    
    def _save_state(self) -> None:
        """Save the current state to PickleDB."""
        try:
            # Update PickleDB state
            self.pickle_db.state.update({
                'traders': self.traders,
                'portfolios': self.portfolios,
                'market_data': self.get_market_prices()
            })
            
            # Save to file
            self.pickle_db.save_state()
        except Exception as e:
            print(f"Error saving system state: {str(e)}")
    
    def get_market_prices(self) -> Dict[str, Any]:
        """Get current market prices from Binance."""
        try:
            prices = {}
            for pair in self.trading_pairs:
                price_data = self.binance.get_current_price(pair)
                if price_data:
                    prices[pair] = {
                        'price': price_data['price'],
                        'change_24h': price_data['change_24h'],
                        'volume': price_data['volume'],
                        'high_24h': price_data['high_24h'],
                        'low_24h': price_data['low_24h']
                    }
            return prices
        except Exception as e:
            print(f"Error getting market prices: {str(e)}")
            return {}
    
    def get_historical_prices(self, symbol: str, interval: str = '1h') -> List[Dict[str, Any]]:
        """Get historical price data for charting."""
        try:
            if symbol not in self.trading_pairs:
                raise ValueError(f"Unsupported trading pair: {symbol}")
            return self.binance.get_historical_prices(symbol, interval)
        except Exception as e:
            print(f"Error getting historical prices: {str(e)}")
            return []
    
    def get_traders(self) -> List[Dict[str, Any]]:
        """Get all traders."""
        return self.traders
    
    def get_portfolio(self, trader_id: str) -> Dict[str, Any]:
        """Get portfolio for a trader."""
        return self.portfolios.get(trader_id, {})
    
    def get_trades(self, trader_id: str) -> List[Dict[str, Any]]:
        """Get trades for a trader."""
        return self.pickle_db.get_trades(trader_id)
    
    def get_agent_portfolio(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio for a specific agent."""
        return self.portfolios.get(agent_id)
    
    def get_trade_history(self, trader_id: str, **kwargs) -> Dict[str, Any]:
        """Get trade history for a trader with filters."""
        # For now, return empty trade history
        return {
            'trades': [],
            'page': 1,
            'total_pages': 1,
            'total_trades': 0
        }
    
    def get_settings(self, section: str) -> Optional[Dict[str, Any]]:
        """Get settings for a specific section."""
        # For now, return None
        return None
    
    def update_settings(self, section: str, data: Dict[str, Any]) -> bool:
        """Update settings for a specific section."""
        # For now, always return success
        return True
    
    def execute_trade(self, trader_id: str, action: str, symbol: str, amount: float) -> Dict[str, Any]:
        """Execute a trade for a trader."""
        try:
            # Get current price
            price_data = self.binance.get_current_price(symbol)
            if not price_data:
                return {'success': False, 'message': 'Failed to get current price'}
            
            price = price_data['price']
            value = amount * price if action == 'buy' else amount / price
            
            # Update portfolio
            portfolio = self.portfolios.get(trader_id, {})
            if not portfolio:
                return {'success': False, 'message': 'Trader not found'}
            
            if action == 'buy':
                if portfolio['base_amount'] < value:
                    return {'success': False, 'message': 'Insufficient funds'}
                
                portfolio['base_amount'] -= value
                portfolio['assets_value'] += value
                portfolio['num_assets'] += 1
                
                # Add to holdings
                holding = {
                    'symbol': symbol,
                    'amount': amount,
                    'value': value,
                    'price': price,
                    'timestamp': datetime.now().isoformat()
                }
                portfolio['holdings'].append(holding)
            else:
                # Find holding to sell
                holding = next((h for h in portfolio['holdings'] if h['symbol'] == symbol), None)
                if not holding:
                    return {'success': False, 'message': 'No holdings found for symbol'}
                
                if holding['amount'] < amount:
                    return {'success': False, 'message': 'Insufficient holdings'}
                
                portfolio['base_amount'] += value
                portfolio['assets_value'] -= value
                portfolio['num_assets'] -= 1
                
                # Update holding
                holding['amount'] -= amount
                holding['value'] -= value
                if holding['amount'] <= 0:
                    portfolio['holdings'].remove(holding)
            
            # Update total value
            portfolio['total_value'] = portfolio['base_amount'] + portfolio['assets_value']
            
            # Save trade
            trade = {
                'trader_id': trader_id,
                'action': action,
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
            self.pickle_db.state['trades'].append(trade)
            
            # Save state
            self._save_state()
            
            return {
                'success': True,
                'message': f"{action.capitalize()} {amount} {symbol} at ${price:.2f}"
            }
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def execute_buy_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a buy order."""
        data['action'] = 'buy'
        return self.execute_trade(data['trader_id'], data['action'], data['symbol'], float(data['amount']))
    
    def execute_sell_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sell order."""
        data['action'] = 'sell'
        return self.execute_trade(data['trader_id'], data['action'], data['symbol'], float(data['amount']))
    
    def get_performance_history(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance metrics history for an agent."""
        # For now, return empty list until we implement performance tracking
        return []
    
    def get_portfolio_history(self, agent_id: str, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get portfolio history for an agent."""
        # For now, return empty list until we implement portfolio history tracking
        return []
    
    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Get all trades from all traders."""
        return self.pickle_db.state.get('trades', [])
    
    def get_signals(self, trader_id: str) -> List[Dict[str, Any]]:
        """Get signals for a specific trader."""
        return [signal for signal in self.pickle_db.state.get('signals', []) 
                if signal.get('trader_id') == trader_id]
    
    def get_all_signals(self) -> List[Dict[str, Any]]:
        """Get all signals from all traders."""
        return self.pickle_db.state.get('signals', [])
    
    def add_signal(self, trader_id: str, signal_data: Dict[str, Any]) -> None:
        """Add a new trading signal."""
        signal = {
            'trader_id': trader_id,
            'timestamp': datetime.now().isoformat(),
            **signal_data
        }
        self.pickle_db.state['signals'].append(signal)
        self._save_state()
    
    def update_signal_status(self, signal_id: str, status: str) -> None:
        """Update the status of a trading signal."""
        signals = self.pickle_db.state.get('signals', [])
        for signal in signals:
            if signal.get('id') == signal_id:
                signal['status'] = status
                break
        self._save_state() 