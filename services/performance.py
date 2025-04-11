"""
Performance Tracking Service - Handles performance tracking and analysis for traders.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: Decimal
    avg_pnl: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_trade_duration: timedelta
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    consecutive_wins: int
    consecutive_losses: int

class PerformanceTracker:
    """Service for tracking and analyzing trader performance."""
    
    def __init__(self, db_handler: Any):
        """
        Initialize the performance tracker.
        
        Args:
            db_handler: Database handler instance
        """
        self.db = db_handler
        self.logger = logging.getLogger('performance_tracker')
        self.logger.info("Performance Tracker initialized")
    
    def calculate_metrics(self, trader_id: str, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> PerformanceMetrics:
        """
        Calculate performance metrics for a trader.
        
        Args:
            trader_id: ID of the trader
            start_time: Optional start time for analysis
            end_time: Optional end time for analysis
            
        Returns:
            PerformanceMetrics object containing calculated metrics
        """
        try:
            # Get trades from database
            trades = self.db.get_trades(trader_id, start_time, end_time)
            
            if not trades:
                return PerformanceMetrics(
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    total_pnl=Decimal('0'),
                    avg_pnl=Decimal('0'),
                    max_drawdown=Decimal('0'),
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    profit_factor=0.0,
                    avg_trade_duration=timedelta(),
                    avg_win=Decimal('0'),
                    avg_loss=Decimal('0'),
                    largest_win=Decimal('0'),
                    largest_loss=Decimal('0'),
                    consecutive_wins=0,
                    consecutive_losses=0
                )
            
            # Convert trades to DataFrame
            df = pd.DataFrame(trades)
            
            # Calculate basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate PnL metrics
            total_pnl = Decimal(str(df['pnl'].sum()))
            avg_pnl = Decimal(str(df['pnl'].mean()))
            
            # Calculate max drawdown
            cumulative_pnl = df['pnl'].cumsum()
            rolling_max = cumulative_pnl.expanding().max()
            drawdowns = cumulative_pnl - rolling_max
            max_drawdown = Decimal(str(abs(drawdowns.min())))
            
            # Calculate risk-adjusted returns
            returns = df['pnl'].pct_change().dropna()
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Calculate profit factor
            gross_profit = df[df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Calculate trade duration metrics
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['duration'] = df['exit_time'] - df['entry_time']
            avg_trade_duration = df['duration'].mean()
            
            # Calculate win/loss metrics
            winning_trades_df = df[df['pnl'] > 0]
            losing_trades_df = df[df['pnl'] < 0]
            
            avg_win = Decimal(str(winning_trades_df['pnl'].mean())) if not winning_trades_df.empty else Decimal('0')
            avg_loss = Decimal(str(losing_trades_df['pnl'].mean())) if not losing_trades_df.empty else Decimal('0')
            largest_win = Decimal(str(winning_trades_df['pnl'].max())) if not winning_trades_df.empty else Decimal('0')
            largest_loss = Decimal(str(losing_trades_df['pnl'].min())) if not losing_trades_df.empty else Decimal('0')
            
            # Calculate consecutive wins/losses
            df['win'] = df['pnl'] > 0
            consecutive_wins = self._calculate_consecutive_wins(df['win'])
            consecutive_losses = self._calculate_consecutive_losses(df['win'])
            
            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_pnl=avg_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def _calculate_consecutive_wins(self, wins: pd.Series) -> int:
        """Calculate maximum consecutive wins."""
        if len(wins) == 0:
            return 0
        
        current_streak = 0
        max_streak = 0
        
        for win in wins:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_consecutive_losses(self, wins: pd.Series) -> int:
        """Calculate maximum consecutive losses."""
        if len(wins) == 0:
            return 0
        
        current_streak = 0
        max_streak = 0
        
        for win in wins:
            if not win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_performance_summary(self, trader_id: str) -> Dict[str, Any]:
        """
        Get a summary of trader performance.
        
        Args:
            trader_id: ID of the trader
            
        Returns:
            Dictionary containing performance summary
        """
        try:
            # Calculate metrics for different time periods
            all_time = self.calculate_metrics(trader_id)
            last_month = self.calculate_metrics(
                trader_id,
                start_time=datetime.now() - timedelta(days=30)
            )
            last_week = self.calculate_metrics(
                trader_id,
                start_time=datetime.now() - timedelta(days=7)
            )
            
            return {
                'all_time': {
                    'total_trades': all_time.total_trades,
                    'win_rate': all_time.win_rate,
                    'total_pnl': float(all_time.total_pnl),
                    'sharpe_ratio': all_time.sharpe_ratio
                },
                'last_month': {
                    'total_trades': last_month.total_trades,
                    'win_rate': last_month.win_rate,
                    'total_pnl': float(last_month.total_pnl),
                    'sharpe_ratio': last_month.sharpe_ratio
                },
                'last_week': {
                    'total_trades': last_week.total_trades,
                    'win_rate': last_week.win_rate,
                    'total_pnl': float(last_week.total_pnl),
                    'sharpe_ratio': last_week.sharpe_ratio
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {str(e)}")
            return {} 