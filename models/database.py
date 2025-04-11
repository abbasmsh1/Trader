"""
Database models using SQLAlchemy ORM.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import json

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

class DatabaseHandler:
    """Database handler using SQLAlchemy."""
    
    def __init__(self, db_url="sqlite:///trading.db"):
        """Initialize database connection."""
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
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
    
    def save_trade(self, agent_id: str, symbol: str, trade_type: str, 
                  quantity: float, price: float, value: float, fee: float, 
                  metadata: dict = None) -> bool:
        """Save trade record to database."""
        try:
            session = self.Session()
            trade = Trade(
                agent_id=agent_id,
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                value=value,
                fee=fee,
                metadata=metadata
            )
            session.add(trade)
            session.commit()
            return True
        except Exception as e:
            print(f"Error saving trade: {str(e)}")
            return False
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