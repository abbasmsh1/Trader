"""
Trader Agents package for Crypto Trader.

This package contains various trader agent implementations,
each with unique trading strategies and personalities.
"""

# Import trader agent classes for easier access
from .base_trader import BaseTraderAgent
from .buffett_trader import BuffettTraderAgent
from .soros_trader import SorosTraderAgent
from .simons_trader import SimonsTraderAgent
from .lynch_trader import LynchTraderAgent

__all__ = [
    'BaseTraderAgent',
    'BuffettTraderAgent',
    'SorosTraderAgent',
    'SimonsTraderAgent',
    'LynchTraderAgent'
] 