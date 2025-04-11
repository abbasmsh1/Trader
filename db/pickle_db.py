"""
Pickle-based database handler for development and testing.
"""

import os
import pickle
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger('pickle_db')

class PickleDBHandler:
    """Database handler using pickle files for storage."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the pickle database handler.
        
        Args:
            data_dir: Directory to store pickle files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data files
        self.traders_file = os.path.join(data_dir, "traders.pickle")
        self.wallets_file = os.path.join(data_dir, "wallets.pickle")
        self.trades_file = os.path.join(data_dir, "trades.pickle")
        self.signals_file = os.path.join(data_dir, "signals.pickle")
        self.agent_states_file = os.path.join(data_dir, "agent_states.pickle")
        
        # Initialize data structures
        self.traders = self._load_or_create(self.traders_file, [
            {
                'id': 'buffett_trader',
                'name': 'Warren Buffett',
                'type': 'value',
                'description': 'Value investor focused on fundamentals'
            },
            {
                'id': 'soros_trader',
                'name': 'George Soros',
                'type': 'macro',
                'description': 'Macro investor focused on reflexivity'
            },
            {
                'id': 'simons_trader',
                'name': 'Jim Simons',
                'type': 'quant',
                'description': 'Quantitative trader using statistical models'
            },
            {
                'id': 'lynch_trader',
                'name': 'Peter Lynch',
                'type': 'growth',
                'description': 'Growth investor looking for tenbaggers'
            }
        ])
        self.wallets = self._load_or_create(self.wallets_file, [])
        self.trades = self._load_or_create(self.trades_file, [])
        self.signals = self._load_or_create(self.signals_file, [])
        self.agent_states = self._load_or_create(self.agent_states_file, {})
        
        logger.info(f"Loaded {len(self.agent_states)} agent states from disk")
    
    def _load_or_create(self, filepath: str, default_data: Any) -> Any:
        """Load data from pickle file or create with default data."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            return default_data
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return default_data
    
    def _save_pickle(self, filepath: str, data: Any) -> bool:
        """Save data to pickle file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving to {filepath}: {str(e)}")
            return False
    
    # Trader operations
    def get_traders(self) -> List[Dict]:
        """Get all traders."""
        return self.traders
    
    def get_trader(self, trader_id: str) -> Optional[Dict]:
        """Get a specific trader."""
        return next((t for t in self.traders if t['id'] == trader_id), None)
    
    def save_trader(self, trader_data: Dict) -> bool:
        """Save trader data."""
        # Update existing or add new
        trader_idx = next((i for i, t in enumerate(self.traders) if t['id'] == trader_data['id']), -1)
        if trader_idx >= 0:
            self.traders[trader_idx] = trader_data
        else:
            self.traders.append(trader_data)
        
        return self._save_pickle(self.traders_file, self.traders)
    
    # Wallet operations
    def get_wallets(self) -> List[Dict]:
        """Get all wallets."""
        return self.wallets
    
    def get_wallet(self, trader_id: str) -> Optional[Dict]:
        """Get a specific wallet."""
        return next((w for w in self.wallets if w['trader_id'] == trader_id), None)
    
    def save_wallet(self, wallet_data: Dict) -> bool:
        """Save wallet data."""
        # Update existing or add new
        wallet_idx = next((i for i, w in enumerate(self.wallets) if w['trader_id'] == wallet_data['trader_id']), -1)
        if wallet_idx >= 0:
            self.wallets[wallet_idx] = wallet_data
        else:
            self.wallets.append(wallet_data)
        
        return self._save_pickle(self.wallets_file, self.wallets)
    
    # Trade operations
    def get_trades(self, trader_id: Optional[str] = None,
                  start_time: Optional[str] = None,
                  end_time: Optional[str] = None,
                  symbol: Optional[str] = None,
                  side: Optional[str] = None,
                  min_amount: Optional[float] = None,
                  page: int = 1,
                  per_page: int = 20) -> Dict[str, Any]:
        """Get trades with filtering and pagination."""
        trades = self.trades.copy()
        
        # Apply filters
        if trader_id:
            trades = [t for t in trades if t['trader_id'] == trader_id]
        if start_time:
            trades = [t for t in trades if t['timestamp'] >= start_time]
        if end_time:
            trades = [t for t in trades if t['timestamp'] <= end_time]
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        if side:
            trades = [t for t in trades if t['side'] == side]
        if min_amount:
            trades = [t for t in trades if float(t['amount']) >= min_amount]
        
        # Sort by timestamp descending
        trades.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Paginate
        total = len(trades)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        trades_page = trades[start_idx:end_idx]
        
        return {
            'trades': trades_page,
            'pagination': {
                'current_page': page,
                'total_pages': (total + per_page - 1) // per_page,
                'total_items': total,
                'per_page': per_page
            }
        }
    
    def save_trade(self, trade_data: Dict) -> bool:
        """Save trade data."""
        self.trades.append(trade_data)
        return self._save_pickle(self.trades_file, self.trades)
    
    # Signal operations
    def get_signals(self, trader_id: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   status: Optional[str] = None,
                   page: int = 1,
                   per_page: int = 20) -> Dict[str, Any]:
        """Get signals with filtering and pagination."""
        signals = self.signals.copy()
        
        # Apply filters
        if trader_id:
            signals = [s for s in signals if s['trader_id'] == trader_id]
        if start_time:
            signals = [s for s in signals if s['timestamp'] >= start_time]
        if end_time:
            signals = [s for s in signals if s['timestamp'] <= end_time]
        if status:
            signals = [s for s in signals if s['status'] == status]
        
        # Sort by timestamp descending
        signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Paginate
        total = len(signals)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        signals_page = signals[start_idx:end_idx]
        
        return {
            'signals': signals_page,
            'pagination': {
                'current_page': page,
                'total_pages': (total + per_page - 1) // per_page,
                'total_items': total,
                'per_page': per_page
            }
        }
    
    def save_signal(self, signal_data: Dict) -> bool:
        """Save signal data."""
        self.signals.append(signal_data)
        return self._save_pickle(self.signals_file, self.signals)
    
    # Agent state operations
    def save_agent_state(self, agent_id: str, state_data: Dict) -> bool:
        """Save agent state."""
        self.agent_states[agent_id] = {
            'state': state_data,
            'updated_at': datetime.now().isoformat()
        }
        return self._save_pickle(self.agent_states_file, self.agent_states)
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict]:
        """Get agent state."""
        return self.agent_states.get(agent_id, {}).get('state') 