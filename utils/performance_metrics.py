"""
Performance metrics utilities for strategy optimization.
"""

import numpy as np
from typing import List

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate of return
        
    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0

def calculate_max_drawdown(returns: List[float]) -> float:
    """
    Calculate the maximum drawdown for a series of returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Maximum drawdown as a percentage
    """
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio for a series of returns.
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate of return
        
    Returns:
        Sortino ratio
    """
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    return np.mean(excess_returns) / downside_std if downside_std != 0 else 0.0

def calculate_calmar_ratio(returns: List[float]) -> float:
    """
    Calculate the Calmar ratio for a series of returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Calmar ratio
    """
    if not returns:
        return 0.0
    
    annualized_return = np.mean(returns) * 252  # Assuming daily returns
    max_dd = calculate_max_drawdown(returns)
    return annualized_return / max_dd if max_dd != 0 else 0.0

def calculate_win_rate(returns: List[float]) -> float:
    """
    Calculate the win rate for a series of returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Win rate as a percentage
    """
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)
    return winning_trades / total_trades if total_trades > 0 else 0.0

def calculate_profit_factor(returns: List[float]) -> float:
    """
    Calculate the profit factor for a series of returns.
    
    Args:
        returns: List of returns
        
    Returns:
        Profit factor
    """
    if not returns:
        return 0.0
    
    returns = np.array(returns)
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    return gross_profit / gross_loss if gross_loss != 0 else float('inf')

def calculate_risk_adjusted_return(returns: List[float]) -> float:
    """
    Calculate a composite risk-adjusted return score.
    
    Args:
        returns: List of returns
        
    Returns:
        Risk-adjusted return score
    """
    if not returns:
        return 0.0
    
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(returns)
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    
    # Normalize profit factor to a reasonable range
    profit_factor = min(profit_factor, 10.0)
    
    # Combine metrics with weights
    score = (
        0.3 * sharpe +
        0.2 * sortino +
        0.2 * calmar +
        0.2 * win_rate +
        0.1 * (profit_factor / 10.0)
    )
    
    return score 