"""
Database handler for file-based storage.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

logger = logging.getLogger('database')

class DatabaseHandler:
    """Handles all database operations using file-based storage."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the database handler.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data files
        self.traders_file = os.path.join(data_dir, "traders.json")
        self.wallets_file = os.path.join(data_dir, "wallets.json")
        self.trades_file = os.path.join(data_dir, "trades.json")
        self.signals_file = os.path.join(data_dir, "signals.json")
        
        # Create files if they don't exist
        self._initialize_files()
        
        logger.info("Database handler initialized")
    
    def _initialize_files(self):
        """Initialize data files if they don't exist."""
        files = {
            self.traders_file: [],
            self.wallets_file: [],
            self.trades_file: [],
            self.signals_file: []
        }
        
        for file_path, default_data in files.items():
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump(default_data, f)
    
    def _load_json(self, file_path: str) -> List[Dict]:
        """Load data from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def _save_json(self, file_path: str, data: List[Dict]) -> bool:
        """Save data to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {str(e)}")
            return False
    
    # Trader operations
    def get_traders(self) -> List[Dict]:
        """Get all traders."""
        return self._load_json(self.traders_file)
    
    def get_trader(self, trader_id: str) -> Optional[Dict]:
        """Get a specific trader."""
        traders = self.get_traders()
        return next((t for t in traders if t['id'] == trader_id), None)
    
    def save_trader(self, trader_data: Dict) -> bool:
        """Save trader data."""
        traders = self.get_traders()
        
        # Update existing or add new
        trader_idx = next((i for i, t in enumerate(traders) if t['id'] == trader_data['id']), -1)
        if trader_idx >= 0:
            traders[trader_idx] = trader_data
        else:
            traders.append(trader_data)
        
        return self._save_json(self.traders_file, traders)
    
    # Wallet operations
    def get_wallets(self) -> List[Dict]:
        """Get all wallets."""
        return self._load_json(self.wallets_file)
    
    def get_wallet(self, trader_id: str) -> Optional[Dict]:
        """Get a specific wallet."""
        wallets = self.get_wallets()
        return next((w for w in wallets if w['trader_id'] == trader_id), None)
    
    def save_wallet(self, wallet_data: Dict) -> bool:
        """Save wallet data."""
        wallets = self.get_wallets()
        
        # Update existing or add new
        wallet_idx = next((i for i, w in enumerate(wallets) if w['trader_id'] == wallet_data['trader_id']), -1)
        if wallet_idx >= 0:
            wallets[wallet_idx] = wallet_data
        else:
            wallets.append(wallet_data)
        
        return self._save_json(self.wallets_file, wallets)
    
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
        trades = self._load_json(self.trades_file)
        
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
        trades = self._load_json(self.trades_file)
        trades.append(trade_data)
        return self._save_json(self.trades_file, trades)
    
    # Signal operations
    def get_signals(self, trader_id: Optional[str] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   status: Optional[str] = None,
                   page: int = 1,
                   per_page: int = 20) -> Dict[str, Any]:
        """Get signals with filtering and pagination."""
        signals = self._load_json(self.signals_file)
        
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
        signals = self._load_json(self.signals_file)
        signals.append(signal_data)
        return self._save_json(self.signals_file, signals)
    
    def save_agent_state(self, agent_id: str, state_data: Dict) -> bool:
        """Save agent state."""
        state = {
            'agent_id': agent_id,
            'state': state_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.save_signal(state)
    
    def load_agent_state(self, agent_id: str) -> Optional[Dict]:
        """Load agent state."""
        signals = self._load_json(self.signals_file)
        states = [s for s in signals if s['agent_id'] == agent_id]
        if states:
            return states[-1]['state']  # Return most recent state
        return None 