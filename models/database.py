"""
Database models using SQLAlchemy ORM.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import json
from typing import Dict, Any, List, Optional
import os
import logging
import pickle

Base = declarative_base()

class AgentState(Base):
    """Model for storing agent states."""
    __tablename__ = 'agent_states'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    agent_type = Column(String, nullable=False)
    state_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Trade(Base):
    """Model for storing trade records."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False)
    trade_type = Column(String, nullable=False)  # buy or sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)  # Additional trade metadata

class Portfolio(Base):
    """Model for storing portfolio snapshots."""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False)
    position = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    last_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PerformanceMetrics(Base):
    """Model for storing performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    profit_sum = Column(Float, default=0.0)
    loss_sum = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    average_win = Column(Float, default=0.0)
    average_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    profit_percentage = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

logger = logging.getLogger('database')

class DatabaseHandler:
    """Handles all database operations."""
    
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
        """
        Get trades with filtering and pagination.
        
        Args:
            trader_id: Filter by trader
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            symbol: Filter by trading pair
            side: Filter by trade side ('buy' or 'sell')
            min_amount: Filter by minimum amount
            page: Page number
            per_page: Items per page
            
        Returns:
            Dict containing trades and pagination info
        """
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
        """
        Get signals with filtering and pagination.
        
        Args:
            trader_id: Filter by trader
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            status: Filter by status
            page: Page number
            per_page: Items per page
            
        Returns:
            Dict containing signals and pagination info
        """
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

    def save_agent_state(self, agent_id: str, agent_type: str, state_data: dict) -> bool:
        """Save agent state to database."""
        try:
            session = self.Session()
            state = AgentState(
                agent_id=agent_id,
                agent_type=agent_type,
                state_data=state_data
            )
            session.add(state)
            session.commit()
            return True
        except Exception as e:
            print(f"Error saving agent state: {str(e)}")
            return False
        finally:
            session.close()
    
    def load_agent_state(self, agent_id: str) -> dict:
        """Load agent state from database."""
        try:
            session = self.Session()
            state = session.query(AgentState).filter_by(agent_id=agent_id).order_by(AgentState.updated_at.desc()).first()
            return state.state_data if state else {}
        except Exception as e:
            print(f"Error loading agent state: {str(e)}")
            return {}
        finally:
            session.close()
    
    def save_portfolio(self, agent_id: str, symbol: str, position: float, 
                      entry_price: float, last_price: float, 
                      unrealized_pnl: float, realized_pnl: float) -> bool:
        """Save portfolio snapshot to database."""
        try:
            session = self.Session()
            portfolio = Portfolio(
                agent_id=agent_id,
                symbol=symbol,
                position=position,
                entry_price=entry_price,
                last_price=last_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl
            )
            session.add(portfolio)
            session.commit()
            return True
        except Exception as e:
            print(f"Error saving portfolio: {str(e)}")
            return False
        finally:
            session.close()
    
    def save_performance_metrics(self, agent_id: str, metrics: dict) -> bool:
        """Save performance metrics to database."""
        try:
            session = self.Session()
            performance = PerformanceMetrics(
                agent_id=agent_id,
                **metrics
            )
            session.add(performance)
            session.commit()
            return True
        except Exception as e:
            print(f"Error saving performance metrics: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_trade_history(self, agent_id: str, limit: int = 100) -> list:
        """Get trade history for an agent."""
        try:
            session = self.Session()
            trades = session.query(Trade).filter_by(agent_id=agent_id).order_by(Trade.timestamp.desc()).limit(limit).all()
            return [{
                'symbol': trade.symbol,
                'type': trade.trade_type,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'fee': trade.fee,
                'timestamp': trade.timestamp.isoformat(),
                'metadata': trade.metadata
            } for trade in trades]
        except Exception as e:
            print(f"Error getting trade history: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_portfolio_history(self, agent_id: str, symbol: str = None, limit: int = 100) -> list:
        """Get portfolio history for an agent."""
        try:
            session = self.Session()
            query = session.query(Portfolio).filter_by(agent_id=agent_id)
            if symbol:
                query = query.filter_by(symbol=symbol)
            portfolios = query.order_by(Portfolio.timestamp.desc()).limit(limit).all()
            return [{
                'symbol': portfolio.symbol,
                'position': portfolio.position,
                'entry_price': portfolio.entry_price,
                'last_price': portfolio.last_price,
                'unrealized_pnl': portfolio.unrealized_pnl,
                'realized_pnl': portfolio.realized_pnl,
                'timestamp': portfolio.timestamp.isoformat()
            } for portfolio in portfolios]
        except Exception as e:
            print(f"Error getting portfolio history: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_performance_history(self, agent_id: str, limit: int = 100) -> list:
        """Get performance metrics history for an agent."""
        try:
            session = self.Session()
            metrics = session.query(PerformanceMetrics).filter_by(agent_id=agent_id).order_by(PerformanceMetrics.timestamp.desc()).limit(limit).all()
            return [{
                'total_trades': metric.total_trades,
                'winning_trades': metric.winning_trades,
                'losing_trades': metric.losing_trades,
                'profit_sum': metric.profit_sum,
                'loss_sum': metric.loss_sum,
                'largest_win': metric.largest_win,
                'largest_loss': metric.largest_loss,
                'win_rate': metric.win_rate,
                'average_win': metric.average_win,
                'average_loss': metric.average_loss,
                'profit_factor': metric.profit_factor,
                'profit_percentage': metric.profit_percentage,
                'timestamp': metric.timestamp.isoformat()
            } for metric in metrics]
        except Exception as e:
            print(f"Error getting performance history: {str(e)}")
            return []
        finally:
            session.close()

    def add_signal(self, trader_id: str, signal_data: Dict[str, Any]) -> None:
        """
        Add a new trading signal to the database.
        
        Args:
            trader_id: ID of the trader
            signal_data: Signal data dictionary
        """
        try:
            # Create signals collection if it doesn't exist
            if "signals" not in self.db.list_collection_names():
                self.db.create_collection("signals")
            
            # Add trader_id to signal data
            signal_data["trader_id"] = trader_id
            
            # Insert signal
            self.db.signals.insert_one(signal_data)
            
        except Exception as e:
            print(f"Error adding signal: {e}")

    def get_active_signals(self, trader_id: str) -> List[Dict[str, Any]]:
        """
        Get active trading signals for a trader.
        
        Args:
            trader_id: ID of the trader
            
        Returns:
            List of active signal dictionaries
        """
        try:
            signals = list(self.db.signals
                          .find({
                              "trader_id": trader_id,
                              "status": "active"
                          })
                          .sort("timestamp", -1))
            
            # Convert ObjectId to string
            for signal in signals:
                signal["_id"] = str(signal["_id"])
                
            return signals
            
        except Exception as e:
            print(f"Error getting active signals: {e}")
            return [] 