import os
import time
import random
import pandas as pd
import numpy as np
import ta
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from threading import Lock, Thread

from data.market_data import MarketDataFetcher
from data.wallet import Wallet

# System parameters
SYSTEM_PARAMS = {
    'update_interval': 60,  # Default update interval in seconds
    'cache_ttl': 300,       # Cache time-to-live in seconds (5 minutes)
    'save_interval': 300,   # State save interval in seconds (5 minutes)
    'record_interval': 60,  # Holdings record interval in seconds (1 minute)
    'max_history': 1000,    # Maximum number of signals to keep in history
    'max_holdings_history': 1000,  # Maximum number of holdings records to keep
    'min_trade_amount_usdt': 3.0,  # Minimum trade amount in USDT (increased to $3)
    'min_trade_amount_crypto': 0.0001  # Minimum trade amount in crypto
}

class TradingSystem:
    def __init__(self, symbols: List[str] = None, goal_filter: str = None):
        """
        Initialize the trading system with personality-based agents.
        
        Args:
            symbols: List of trading pairs to analyze
            goal_filter: Filter agents by goal type ('usd' or 'btc')
        """
        self.data_fetcher = MarketDataFetcher()
        
        # Default symbols including USDT pairs and coin-to-coin pairs
        default_symbols = [
            # Major coins
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
            
            # Coin-to-coin pairs
            'BTC/ETH', 'ETH/BTC', 'BNB/BTC', 'SOL/BTC',
            
            # AI and Gaming tokens
            'ARKM/USDT', 'AGIX/USDT', 'IMX/USDT', 'RNDR/USDT',
            
            # Meme coins
            'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT',
            
            # Alt coins
            'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT', 'XRP/USDT',
            'ATOM/USDT', 'UNI/USDT', 'AAVE/USDT', 'NEAR/USDT', 'ARB/USDT'
        ]
        
        self.symbols = symbols or default_symbols
        
        # Initialize trading agents with different personalities
        all_agents = [
            {
                'name': 'Warren Buffett AI',
                'personality': 'Value Investor',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.3,
                'strategy': {
                    'value_investing': 0.9,
                    'momentum_trading': 0.1,
                    'trend_following': 0.2,
                    'mean_reversion': 0.5,
                    'scalping': 0.0,
                    'swing_trading': 0.3
                },
                'goal': 'usd',  # Goal is to reach $100
                'market_view': {
                    'market_trend': 'neutral',
                    'risk_assessment': 'conservative',
                    'time_horizon': 'very_long',
                    'sentiment': 'cautious'
                },
                'preferred_pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                'avoid_pairs': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT']
            },
            {
                'name': 'Elon Musk AI',
                'personality': 'Tech Disruptor',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.8,
                'strategy': {
                    'value_investing': 0.2,
                    'momentum_trading': 0.9,
                    'trend_following': 0.7,
                    'mean_reversion': 0.1,
                    'scalping': 0.6,
                    'swing_trading': 0.4
                },
                'goal': 'usd',  # Goal is to reach $100
                'market_view': {
                    'market_trend': 'extremely_bullish',
                    'risk_assessment': 'high_reward',
                    'time_horizon': 'medium',
                    'sentiment': 'optimistic'
                },
                'preferred_pairs': ['DOGE/USDT', 'SHIB/USDT', 'RNDR/USDT', 'AGIX/USDT'],
                'avoid_pairs': ['XRP/USDT', 'ADA/USDT']
            },
            {
                'name': 'Michael Burry AI',
                'personality': 'Contrarian',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.6,
                'strategy': {
                    'value_investing': 0.7,
                    'momentum_trading': 0.2,
                    'trend_following': 0.1,
                    'mean_reversion': 0.9,
                    'scalping': 0.3,
                    'swing_trading': 0.5
                },
                'goal': 'usd',  # Goal is to reach $100
                'market_view': {
                    'market_trend': 'bearish',
                    'risk_assessment': 'bubble_warning',
                    'time_horizon': 'short',
                    'sentiment': 'pessimistic'
                },
                'preferred_pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                'avoid_pairs': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
            },
            {
                'name': 'Ray Dalio AI',
                'personality': 'Macro Trader',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.5,
                'strategy': {
                    'value_investing': 0.6,
                    'momentum_trading': 0.4,
                    'trend_following': 0.8,
                    'mean_reversion': 0.3,
                    'scalping': 0.1,
                    'swing_trading': 0.5
                },
                'goal': 'usd',  # Goal is to reach $100
                'market_view': {
                    'market_trend': 'cyclical',
                    'risk_assessment': 'balanced',
                    'time_horizon': 'long',
                    'sentiment': 'analytical'
                },
                'preferred_pairs': ['BTC/USDT', 'ETH/USDT', 'LINK/USDT', 'DOT/USDT'],
                'avoid_pairs': []
            },
            {
                'name': 'Jesse Livermore AI',
                'personality': 'Swing Trader',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.8,
                'strategy': {
                    'value_investing': 0.1,
                    'momentum_trading': 0.7,
                    'trend_following': 0.6,
                    'mean_reversion': 0.4,
                    'scalping': 0.5,
                    'swing_trading': 0.9
                },
                'goal': 'usd',  # Goal is to reach $100
                'market_view': {
                    'market_trend': 'volatile',
                    'risk_assessment': 'opportunity',
                    'time_horizon': 'short',
                    'sentiment': 'opportunistic'
                },
                'preferred_pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT'],
                'avoid_pairs': []
            },
            # BTC Goal Agents - Same personalities but with BTC accumulation goal
            {
                'name': 'Satoshi AI',
                'personality': 'Bitcoin Maximalist',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.4,
                'strategy': {
                    'value_investing': 0.8,
                    'momentum_trading': 0.3,
                    'trend_following': 0.4,
                    'mean_reversion': 0.5,
                    'scalping': 0.1,
                    'swing_trading': 0.2
                },
                'goal': 'btc',  # Goal is to reach 1 BTC
                'market_view': {
                    'market_trend': 'long_term_bullish',
                    'risk_assessment': 'moderate',
                    'time_horizon': 'very_long',
                    'sentiment': 'conviction'
                },
                'preferred_pairs': ['BTC/USDT'],
                'avoid_pairs': ['ETH/USDT', 'BNB/USDT', 'SOL/USDT']
            },
            {
                'name': 'Vitalik AI',
                'personality': 'Tech Innovator',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.7,
                'strategy': {
                    'value_investing': 0.4,
                    'momentum_trading': 0.6,
                    'trend_following': 0.5,
                    'mean_reversion': 0.3,
                    'scalping': 0.4,
                    'swing_trading': 0.5
                },
                'goal': 'btc',  # Goal is to reach 1 BTC
                'market_view': {
                    'market_trend': 'innovative',
                    'risk_assessment': 'calculated',
                    'time_horizon': 'medium',
                    'sentiment': 'progressive'
                },
                'preferred_pairs': ['ETH/BTC', 'SOL/BTC', 'LINK/BTC'],
                'avoid_pairs': ['DOGE/USDT', 'SHIB/USDT']
            },
            {
                'name': 'Saylor AI',
                'personality': 'BTC Accumulator',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.9,
                'strategy': {
                    'value_investing': 0.7,
                    'momentum_trading': 0.4,
                    'trend_following': 0.3,
                    'mean_reversion': 0.6,
                    'scalping': 0.2,
                    'swing_trading': 0.3
                },
                'goal': 'btc',  # Goal is to reach 1 BTC
                'market_view': {
                    'market_trend': 'extremely_bullish',
                    'risk_assessment': 'all_in',
                    'time_horizon': 'infinite',
                    'sentiment': 'maximalist'
                },
                'preferred_pairs': ['BTC/USDT'],
                'avoid_pairs': ['ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
            },
            {
                'name': 'Woo AI',
                'personality': 'Data Scientist',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.6,
                'strategy': {
                    'value_investing': 0.5,
                    'momentum_trading': 0.6,
                    'trend_following': 0.7,
                    'mean_reversion': 0.4,
                    'scalping': 0.3,
                    'swing_trading': 0.5
                },
                'goal': 'btc',  # Goal is to reach 1 BTC
                'market_view': {
                    'market_trend': 'data_driven',
                    'risk_assessment': 'statistical',
                    'time_horizon': 'variable',
                    'sentiment': 'analytical'
                },
                'preferred_pairs': ['BTC/USDT', 'ETH/BTC', 'SOL/BTC'],
                'avoid_pairs': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT']
            },
            {
                'name': 'PlanB AI',
                'personality': 'Stock-to-Flow Believer',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.8,
                'strategy': {
                    'value_investing': 0.3,
                    'momentum_trading': 0.5,
                    'trend_following': 0.6,
                    'mean_reversion': 0.4,
                    'scalping': 0.6,
                    'swing_trading': 0.7
                },
                'goal': 'btc',  # Goal is to reach 1 BTC
                'market_view': {
                    'market_trend': 'halving_cycles',
                    'risk_assessment': 'model_based',
                    'time_horizon': 'four_year_cycles',
                    'sentiment': 'mathematical'
                },
                'preferred_pairs': ['BTC/USDT'],
                'avoid_pairs': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
            },
            # Predictive Chart Reading Agent
            {
                'name': 'Oracle AI',
                'personality': 'Chart Reader',
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': 0.5,
                'strategy': {
                    'value_investing': 0.2,
                    'momentum_trading': 0.7,
                    'trend_following': 0.8,
                    'mean_reversion': 0.6,
                    'scalping': 0.5,
                    'swing_trading': 0.6,
                    'pattern_recognition': 0.9  # New strategy component
                },
                'goal': 'usd',  # Goal is to reach $100
                'market_view': {
                    'market_trend': 'pattern_based',
                    'risk_assessment': 'technical',
                    'time_horizon': 'adaptive',
                    'sentiment': 'objective'
                },
                'preferred_pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'AVAX/USDT'],
                'avoid_pairs': []
            }
        ]
        
        # Filter agents by goal if specified
        if goal_filter:
            self.agents = [agent for agent in all_agents if agent.get('goal') == goal_filter]
        else:
            self.agents = all_agents
        
        # Initialize market data cache
        self.market_data_cache = {}
        self.last_update_time = {}
        self.auto_trading_enabled = True
        self.discussions = []
        
        # Initialize signals history
        self.signals_history = []
        
        # Initialize holdings history for each trader
        self.holdings_history = {agent['name']: [] for agent in self.agents}
        
        # Thread safety
        self.lock = Lock()
        
<<<<<<< HEAD
        # Load previous state if available
        self._load_state()
        
        # Make initial BTC purchase for agents with BTC goal
        self._make_initial_btc_purchase()
        
    def _reset_agents(self):
        """Reset all agents to their initial state."""
        print("Resetting all agents to initial state...")
        
        # Store the original goal and personality for each agent
        agent_configs = []
        for agent in self.agents:
            agent_configs.append({
                'name': agent['name'],
                'personality': agent['personality'],
                'goal': agent.get('goal', 'usd'),
                'risk_tolerance': agent.get('risk_tolerance', 0.5),
                'preferred_pairs': agent.get('preferred_pairs', []),
                'avoid_pairs': agent.get('avoid_pairs', [])
            })
        
        # Re-initialize agents with fresh wallets
        self.agents = []
        for config in agent_configs:
            self.agents.append({
                'name': config['name'],
                'personality': config['personality'],
                'wallet': Wallet(initial_balance_usdt=20.0),
                'risk_tolerance': config['risk_tolerance'],
                'strategy': self.get_default_strategy(config['personality']),
                'goal': config['goal'],
                'market_view': self.get_default_market_view(config['personality']),
                'preferred_pairs': config['preferred_pairs'],
                'avoid_pairs': config['avoid_pairs']
            })
        
        # Reset signals history and holdings history
        self.signals_history = []
        self.holdings_history = {agent['name']: [] for agent in self.agents}
        self.discussions = []
        
        # Make initial BTC purchase for agents with BTC goal
        self._make_initial_btc_purchase()
        
        print(f"Reset complete. {len(self.agents)} agents initialized with $20.0 USDT each.")
    
    def get_default_strategy(self, personality: str) -> Dict:
        """Get default strategy based on personality."""
        if 'Value Investor' in personality:
            return {
                'value_investing': 0.9,
                'momentum_trading': 0.1,
                'trend_following': 0.2,
                'mean_reversion': 0.5,
                'scalping': 0.0,
                'swing_trading': 0.3
            }
        elif 'Tech Disruptor' in personality or 'Tech Innovator' in personality:
            return {
                'value_investing': 0.2,
                'momentum_trading': 0.9,
                'trend_following': 0.7,
                'mean_reversion': 0.1,
                'scalping': 0.6,
                'swing_trading': 0.4
            }
        elif 'Contrarian' in personality:
            return {
                'value_investing': 0.7,
                'momentum_trading': 0.2,
                'trend_following': 0.1,
                'mean_reversion': 0.9,
                'scalping': 0.3,
                'swing_trading': 0.5
            }
        elif 'Macro Trader' in personality:
            return {
                'value_investing': 0.6,
                'momentum_trading': 0.4,
                'trend_following': 0.8,
                'mean_reversion': 0.3,
                'scalping': 0.1,
                'swing_trading': 0.5
            }
        elif 'Swing Trader' in personality:
            return {
                'value_investing': 0.1,
                'momentum_trading': 0.7,
                'trend_following': 0.6,
                'mean_reversion': 0.4,
                'scalping': 0.5,
                'swing_trading': 0.9
            }
        elif 'Bitcoin Maximalist' in personality or 'BTC Accumulator' in personality:
            return {
                'value_investing': 0.7,
                'momentum_trading': 0.4,
                'trend_following': 0.3,
                'mean_reversion': 0.6,
                'scalping': 0.2,
                'swing_trading': 0.3
            }
        elif 'Chart Reader' in personality:
            return {
                'value_investing': 0.2,
                'momentum_trading': 0.7,
                'trend_following': 0.8,
                'mean_reversion': 0.6,
                'scalping': 0.5,
                'swing_trading': 0.6,
                'pattern_recognition': 0.9
            }
        else:
            return {
                'value_investing': 0.5,
                'momentum_trading': 0.5,
                'trend_following': 0.5,
                'mean_reversion': 0.5,
                'scalping': 0.5,
                'swing_trading': 0.5
            }
    
    def get_default_market_view(self, personality: str) -> Dict:
        """Get default market view based on personality."""
        if 'Value Investor' in personality:
            return {
                'market_trend': 'neutral',
                'risk_assessment': 'conservative',
                'time_horizon': 'very_long',
                'sentiment': 'cautious'
            }
        elif 'Tech Disruptor' in personality:
            return {
                'market_trend': 'extremely_bullish',
                'risk_assessment': 'high_reward',
                'time_horizon': 'medium',
                'sentiment': 'optimistic'
            }
        elif 'Tech Innovator' in personality:
            return {
                'market_trend': 'innovative',
                'risk_assessment': 'calculated',
                'time_horizon': 'medium',
                'sentiment': 'progressive'
            }
        elif 'Contrarian' in personality:
            return {
                'market_trend': 'bearish',
                'risk_assessment': 'bubble_warning',
                'time_horizon': 'short',
                'sentiment': 'pessimistic'
            }
        elif 'Macro Trader' in personality:
            return {
                'market_trend': 'cyclical',
                'risk_assessment': 'balanced',
                'time_horizon': 'long',
                'sentiment': 'analytical'
            }
        elif 'Swing Trader' in personality:
            return {
                'market_trend': 'volatile',
                'risk_assessment': 'opportunity',
                'time_horizon': 'short',
                'sentiment': 'opportunistic'
            }
        elif 'Bitcoin Maximalist' in personality:
            return {
                'market_trend': 'long_term_bullish',
                'risk_assessment': 'moderate',
                'time_horizon': 'very_long',
                'sentiment': 'conviction'
            }
        elif 'BTC Accumulator' in personality:
            return {
                'market_trend': 'extremely_bullish',
                'risk_assessment': 'all_in',
                'time_horizon': 'infinite',
                'sentiment': 'maximalist'
            }
        elif 'Chart Reader' in personality:
            return {
                'market_trend': 'pattern_based',
                'risk_assessment': 'technical',
                'time_horizon': 'adaptive',
                'sentiment': 'objective'
            }
        else:
            return {
                'market_trend': 'neutral',
                'risk_assessment': 'moderate',
                'time_horizon': 'medium',
                'sentiment': 'balanced'
            }
    
=======
        # Initialize wallets
        for agent in self.agents:
            agent.wallet = Wallet(initial_balance_usdt=20.0)
            agent.wallet.holdings = {}
            agent.wallet.trades_history = []
        
        self.discussions = []  # Store agent discussions
        
        # Load saved state if available
        self._load_state()
        
>>>>>>> 59355cecc9b582561f2aaf98fa89d36d9d48ea41
    def _load_state(self):
        """Load saved state from disk if available."""
        try:
            state_file = os.path.join(os.path.dirname(__file__), 'data', 'trading_state.pkl')
            if os.path.exists(state_file):
                import pickle
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    
                # Restore signals history
                self.signals_history = state.get('signals_history', [])
                
                # Restore agent wallets
                for i, agent_state in enumerate(state.get('agents', [])):
                    if i < len(self.agents):
                        self.agents[i]['wallet'].balance_usdt = agent_state.get('balance_usdt', 20.0)
                        self.agents[i]['wallet'].holdings = agent_state.get('holdings', {})
                        self.agents[i]['wallet'].trades_history = agent_state.get('trades_history', [])
                
                print(f"Loaded saved state with {len(self.signals_history)} signals and {len(state.get('agents', []))} agent wallets")
            else:
                print("No saved state found, starting fresh")
        except Exception as e:
            print(f"Error loading saved state: {str(e)}")
            
    def _save_state(self):
        """Save current state to disk."""
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            state_file = os.path.join(data_dir, 'trading_state.pkl')
            
            # Prepare state to save
            state = {
                'signals_history': self.signals_history,
                'agents': [
                    {
                        'name': agent['name'],
                        'balance_usdt': agent['wallet'].balance_usdt,
                        'holdings': agent['wallet'].holdings,
                        'trades_history': agent['wallet'].trades_history
                    }
                    for agent in self.agents
                ],
                'timestamp': datetime.now()
            }
            
            # Save to file
            import pickle
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
                
            print(f"Saved trading state to {state_file}")
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            
    def _make_initial_btc_purchase(self):
        """Make initial BTC purchase for agents with BTC goal."""
        try:
            # Get current BTC price
            btc_data = self.get_market_data("BTC/USDT")
            if btc_data.empty:
                print("Could not get BTC price data for initial purchase")
                return
            
            btc_price = float(btc_data['close'].iloc[-1])
            print(f"Current BTC price: ${btc_price:.2f}")
            
            # Each agent with BTC goal buys BTC with 80% of their initial balance
            for agent in self.agents:
                # Only make initial BTC purchase for agents with BTC goal
                if agent.get('goal', 'usd') == 'btc':
                    initial_usdt = agent['wallet'].balance_usdt
                    purchase_amount = initial_usdt * 0.8  # Use 80% of initial balance
                    
                    # Ensure minimum purchase amount of $3
                    purchase_amount = max(purchase_amount, 3.0)
                    
                    # Skip if not enough balance
                    if purchase_amount > agent['wallet'].balance_usdt:
                        print(f"❌ {agent['name']} has insufficient balance for initial BTC purchase")
                        continue
                    
                    success = agent['wallet'].execute_buy("BTC/USDT", purchase_amount, btc_price)
                    if success:
                        btc_amount = purchase_amount / btc_price
                        print(f"🔄 {agent['name']} made initial BTC purchase: {btc_amount:.8f} BTC (${purchase_amount:.2f})")
                        
                        # Get personality traits
                        personality_traits = self.get_personality_traits(agent['personality'])
                        
                        # Add to signals history
                        self.signals_history.append({
                            'agent': agent['name'],
                            'personality': personality_traits['personality'],
                            'symbol': "BTC/USDT",
                            'signal': {
                                'action': 'BUY',
                                'confidence': 0.9,
                                'timestamp': datetime.now().timestamp(),
                                'reason': 'Initial purchase to accumulate BTC'
                            },
                            'risk_tolerance': agent.get('risk_tolerance', 0.5),
                            'strategy': personality_traits,
                            'market_view': personality_traits.get('market_beliefs', {}),
                            'wallet_metrics': agent['wallet'].get_performance_metrics({'BTC/USDT': btc_price}),
                            'trade_executed': True,
                            'timestamp': datetime.now().timestamp(),
                            'goal': 'btc'
                        })
                    else:
                        print(f"❌ {agent['name']} failed to make initial BTC purchase")
        except Exception as e:
            print(f"Error making initial BTC purchase: {str(e)}")
            
    def _set_aggressive_strategies(self):
        """Set more aggressive trading strategies for all agents to reach $100 goal faster."""
        for agent in self.agents:
            # Update strategy preferences to be more aggressive and responsive
            agent['strategy'] = {
                'value_investing': 0.3,
                'momentum_trading': 0.8,
                'trend_following': 0.9,
                'swing_trading': 0.7,
                'scalping': 0.6,
                'mean_reversion': 0.7  # Added for better responsiveness to market conditions
            }
            
            # Get current market data for BTC to determine overall market sentiment
            btc_data = self.get_market_data("BTC/USDT")
            if not btc_data.empty:
                # Calculate basic indicators
                rsi = self.calculate_rsi(btc_data['close'])
                sma20 = btc_data['close'].rolling(window=20).mean().iloc[-1]
                sma50 = btc_data['close'].rolling(window=50).mean().iloc[-1]
                
                # Determine market trend based on indicators
                if rsi > 60 and sma20 > sma50:
                    # Bullish market
                    agent['market_view'] = {
                        'market_trend': 'bullish',
                        'volatility_expectation': 'high',
                        'risk_assessment': 'opportunity'
                    }
                elif rsi < 40 and sma20 < sma50:
                    # Bearish market
                    agent['market_view'] = {
                        'market_trend': 'bearish',
                        'volatility_expectation': 'high',
                        'risk_assessment': 'cautious'
                    }
                else:
                    # Neutral market
                    agent['market_view'] = {
                        'market_trend': 'neutral',
                        'volatility_expectation': 'moderate',
                        'risk_assessment': 'balanced'
                    }
            else:
                # Default to balanced view if no data
                agent['market_view'] = {
                    'market_trend': 'neutral',
                    'volatility_expectation': 'moderate',
                    'risk_assessment': 'balanced'
                }
            
            # Adjust risk tolerance based on current holdings
            wallet_metrics = agent['wallet'].get_performance_metrics({})
            total_value = wallet_metrics['total_value_usdt']
            
            if total_value < 30:
                # More aggressive for smaller portfolios
                agent['risk_tolerance'] = min(0.8, agent['risk_tolerance'] * 1.5)
            elif total_value > 70:
                # More conservative for larger portfolios
                agent['risk_tolerance'] = max(0.3, agent['risk_tolerance'] * 0.8)
                
            print(f"Set aggressive strategy for {agent['name']}: {agent['market_view']['market_trend']} outlook, {agent['risk_tolerance']:.2f} risk tolerance")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS
            rs = avg_gain / avg_loss
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return 50  # Return neutral RSI on error
    
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for a symbol, using cache if available."""
        try:
            with self.lock:
                current_time = time.time()
                cache_ttl = SYSTEM_PARAMS['cache_ttl']
                
                # Check if we have cached data that's still fresh
                if (symbol in self.market_data_cache and 
                    symbol in self.last_update_time and 
                    current_time - self.last_update_time[symbol] < cache_ttl):
                    print(f"Using cached data for {symbol}")
                    return self.market_data_cache[symbol]
                
                # Fetch new data
                print(f"Fetching new data for {symbol}")
                
                # Try to use fetch_ohlcv if it exists, otherwise fall back to fetch_market_data
                try:
                    if hasattr(self.data_fetcher, 'fetch_ohlcv'):
                        df = self.data_fetcher.fetch_ohlcv(symbol, timeframe='1h', limit=100)
                    elif hasattr(self.data_fetcher, 'fetch_market_data'):
                        df = self.data_fetcher.fetch_market_data(symbol)
                    else:
                        # Generic fallback - create dummy data for testing
                        print(f"Warning: No method found to fetch data for {symbol}. Using dummy data.")
                        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
                        df = pd.DataFrame({
                            'open': np.random.normal(100, 10, 100),
                            'high': np.random.normal(105, 10, 100),
                            'low': np.random.normal(95, 10, 100),
                            'close': np.random.normal(100, 10, 100),
                            'volume': np.random.normal(1000000, 500000, 100)
                        }, index=dates)
                except Exception as e:
                    print(f"Error fetching data with primary method: {str(e)}. Using fallback.")
                    # Fallback - create dummy data for testing
                    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
                    df = pd.DataFrame({
                        'open': np.random.normal(100, 10, 100),
                        'high': np.random.normal(105, 10, 100),
                        'low': np.random.normal(95, 10, 100),
                        'close': np.random.normal(100, 10, 100),
                        'volume': np.random.normal(1000000, 500000, 100)
                    }, index=dates)
                
                if not df.empty:
                    self.market_data_cache[symbol] = df
                    self.last_update_time[symbol] = current_time
                
                return df
        except Exception as e:
            print(f"Error in get_market_data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def generate_discussion(self, signals: List[Dict]) -> str:
        """Generate a dynamic discussion between agents about their trading signals with competitive banter."""
        if not signals:
            return ""
        
        try:
            # Group signals by goal type
            usd_goal_agents = [s for s in signals if s.get('goal', 'usd') == 'usd']
            btc_goal_agents = [s for s in signals if s.get('goal', 'usd') == 'btc']
            
            # Sort agents by their progress toward their respective goals
            sorted_usd_agents = sorted(usd_goal_agents, key=lambda x: x['wallet_metrics']['total_value_usdt'], reverse=True) if usd_goal_agents else []
            
            # For BTC goal agents, sort by BTC holdings
            sorted_btc_agents = []
            if btc_goal_agents:
                for agent in btc_goal_agents:
                    metrics = agent['wallet_metrics']
                    btc_holdings = 0
                    for symbol, holding in metrics['holdings'].items():
                        if symbol == 'BTC/USDT' or symbol == 'BTC':
                            btc_holdings += holding['amount']
                    agent['btc_holdings'] = btc_holdings
                sorted_btc_agents = sorted(btc_goal_agents, key=lambda x: x.get('btc_holdings', 0), reverse=True)
            
            # Find the Oracle AI agent for predictions
            oracle_agent = next((s for s in signals if s['agent'] == 'Oracle AI'), None)
            
            # Group signals by action type
            bullish_agents = [s for s in signals if s['signal']['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']]
            bearish_agents = [s for s in signals if s['signal']['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT']]
            neutral_agents = [s for s in signals if s['signal']['action'] in ['HOLD', 'WATCH']]
            
            symbol = signals[0]['symbol']
            discussion = []
            
            # Get market data for analysis
            df = self.get_market_data(symbol)
            if not df.empty:
                current_price = float(df['close'].iloc[-1])
                prev_price = float(df['close'].iloc[-2])
                price_change = ((current_price - prev_price) / prev_price) * 100
                volume = float(df['volume'].iloc[-1])
                
                # Calculate technical indicators for deeper analysis
                sma20 = df['close'].rolling(window=20).mean().iloc[-1]
                sma50 = df['close'].rolling(window=50).mean().iloc[-1]
                rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
                trend = "bullish" if sma20 > sma50 else "bearish"
                
                # Calculate additional indicators for Oracle AI
                macd = ta.trend.macd_diff(df['close']).iloc[-1]
                macd_prev = ta.trend.macd_diff(df['close']).iloc[-2] if len(df) > 2 else 0
                bollinger_bands = ta.volatility.bollinger_hband_indicator(df['close']).iloc[-1]
                stoch = ta.momentum.stoch(df['high'], df['low'], df['close']).iloc[-1]
                
                # Start with USD race leader update
                if sorted_usd_agents:
                    usd_leader = sorted_usd_agents[0]
                    usd_leader_value = usd_leader['wallet_metrics']['total_value_usdt']
                    usd_leader_progress = (usd_leader_value / 100.0) * 100
                    
                    discussion.append(
                        f"{usd_leader['agent']}: Leading the $100 race at ${usd_leader_value:.2f} ({usd_leader_progress:.1f}%). "
                        f"Let's analyze {symbol} at ${current_price:.4f}. "
                        f"Market shows a {price_change:.1f}% move with ${volume:,.0f} volume."
                    )
                
                # Add BTC race leader update
                if sorted_btc_agents:
                    btc_leader = sorted_btc_agents[0]
                    btc_holdings = btc_leader.get('btc_holdings', 0)
                    btc_progress = (btc_holdings / 1.0) * 100  # Progress toward 1 BTC
                    
                    discussion.append(
                        f"{btc_leader['agent']}: Leading the 1 BTC race with {btc_holdings:.8f} BTC ({btc_progress:.1f}%). "
                        f"My strategy for {symbol} is focused on accumulating more BTC."
                    )
                
                # Add Oracle AI prediction if available
                if oracle_agent:
                    # Generate chart pattern prediction
                    prediction_direction = "up" if (macd > 0 and rsi < 70 and stoch < 80) else "down"
                    confidence = random.uniform(0.65, 0.95)  # Random confidence between 65-95%
                    
                    # Generate pattern recognition insights
                    patterns = []
                    if rsi > 70:
                        patterns.append("overbought conditions")
                    elif rsi < 30:
                        patterns.append("oversold conditions")
                    
                    if macd > 0 and macd > macd_prev:
                        patterns.append("bullish MACD crossover")
                    elif macd < 0 and macd < macd_prev:
                        patterns.append("bearish MACD divergence")
                    
                    if bollinger_bands > 0:
                        patterns.append("upper Bollinger Band test")
                    
                    pattern_text = ", ".join(patterns) if patterns else "consolidation pattern"
                    
                    # Add Oracle's prediction to discussion
                    discussion.append(
                        f"{oracle_agent['agent']}: My chart analysis predicts {symbol} will go {prediction_direction} with {confidence:.1%} confidence. "
                        f"I'm detecting {pattern_text}. RSI: {rsi:.1f}, MACD: {macd:.6f}, Stochastic: {stoch:.1f}. "
                        f"The next price target is ${current_price * (1.05 if prediction_direction == 'up' else 0.95):.4f}."
                    )
                
                # Generate personality-based market insights with competitive edge
                def get_competitive_response(agent_data, context):
                    personality = agent_data['personality'].lower()
                    metrics = agent_data['wallet_metrics']
                    
                    # Determine goal type and progress
                    goal_type = agent_data.get('goal', 'usd')
                    if goal_type == 'usd':
                        progress = (metrics['total_value_usdt'] / 100.0) * 100
                        progress_text = f"${metrics['total_value_usdt']:.2f} ({progress:.1f}% to $100)"
                    else:  # BTC goal
                        btc_holdings = agent_data.get('btc_holdings', 0)
                        progress = (btc_holdings / 1.0) * 100
                        progress_text = f"{btc_holdings:.8f} BTC ({progress:.1f}% to 1 BTC)"
                    
                    if 'value' in personality:
                        return (f"My value investing approach has grown my portfolio to {progress_text}. "
                               f"The current market valuation {'supports' if context == 'bullish' else 'contradicts'} my thesis.")
                    elif 'tech' in personality:
                        return (f"With {progress_text} and rising, my technical analysis is proving effective. "
                               f"The indicators {'confirm' if context == 'bullish' else 'challenge'} this position.")
                    elif 'contrarian' in personality:
                        return (f"While others follow the crowd, I've built {progress_text} by being contrarian. "
                               f"The market sentiment is too {'optimistic' if context == 'bearish' else 'pessimistic'}.")
                    elif 'macro' in personality:
                        return (f"My macro strategy has accumulated {progress_text}. "
                               f"Current conditions {'favor' if context == 'bullish' else 'discourage'} this position.")
                    elif 'swing' in personality:
                        return (f"Swing trading has grown my account to {progress_text}. "
                               f"The patterns {'support' if context == 'bullish' else 'do not support'} my strategy.")
                    elif 'chart' in personality:
                        return (f"My pattern recognition has built {progress_text}. "
                               f"The chart patterns clearly indicate this market will go {'up' if context == 'bullish' else 'down'}.")
                    else:
                        return (f"With {progress_text} in my portfolio, "
                               f"my analysis {'confirms' if context == 'bullish' else 'contradicts'} this view.")
                
                # Add competitive responses from USD race runner-up
                if len(sorted_usd_agents) > 1:
                    runner_up = sorted_usd_agents[1]
                    runner_up_value = runner_up['wallet_metrics']['total_value_usdt']
                    value_difference = sorted_usd_agents[0]['wallet_metrics']['total_value_usdt'] - runner_up_value
                    discussion.append(
                        f"{runner_up['agent']}: Only ${value_difference:.2f} behind in the $100 race! "
                        f"My {runner_up['personality']} approach will prove superior. "
                        f"This {symbol} setup is perfect for my strategy."
                    )
                
                # Add competitive responses from BTC race runner-up
                if len(sorted_btc_agents) > 1:
                    btc_runner_up = sorted_btc_agents[1]
                    btc_runner_up_holdings = btc_runner_up.get('btc_holdings', 0)
                    btc_difference = sorted_btc_agents[0].get('btc_holdings', 0) - btc_runner_up_holdings
                    discussion.append(
                        f"{btc_runner_up['agent']}: Just {btc_difference:.8f} BTC behind in the 1 BTC race! "
                        f"My {btc_runner_up['personality']} approach is optimized for BTC accumulation. "
                        f"Watch me close the gap with this {symbol} trade."
                    )
                
                # Add competitive bullish perspectives
                if bullish_agents:
                    for bull in bullish_agents[:2]:
                        confidence = bull['signal']['confidence']
                        reason = bull['signal'].get('reason', '')
                        
                        bull_analysis = [
                            f"{bull['agent']}: Watch and learn! This is how you reach your goal first.",
                            get_competitive_response(bull, 'bullish'),
                            f"I'm {bull['signal']['action']} with {confidence:.0%} confidence."
                        ]
                        
                        if reason:
                            bull_analysis.append(f"Key insight: {reason}")
                        
                        discussion.append(" ".join(bull_analysis))
                        
                        # Generate heated counter-arguments
                        if bearish_agents:
                            bear = bearish_agents[0]
                            counter = [
                                f"{bear['agent']}: Bold words for someone with your portfolio size!",
                                get_competitive_response(bear, 'bearish'),
                                f"My analysis is based on data, not wishful thinking."
                            ]
                            discussion.append(" ".join(counter))
                            
                            # Bull defends with competitive spirit
                            defense = [
                                f"{bull['agent']}: We'll see who reaches their goal first! The momentum is clear:",
                                f"RSI at {rsi:.1f}, volume spiking, and my strategy is perfectly positioned."
                            ]
                            discussion.append(" ".join(defense))
                
                # Add strategic neutral perspectives
                if neutral_agents:
                    neutral = neutral_agents[0]
                    
                    neutral_analysis = [
                        f"{neutral['agent']}: While you're all arguing, I'm steadily progressing.",
                        get_competitive_response(neutral, 'neutral')
                    ]
                    
                    # Check if close to leader in respective race
                    if neutral.get('goal', 'usd') == 'usd' and sorted_usd_agents:
                        usd_leader_value = sorted_usd_agents[0]['wallet_metrics']['total_value_usdt']
                        neutral_value = neutral['wallet_metrics']['total_value_usdt']
                        if neutral_value > usd_leader_value * 0.9:  # Close to leader
                            neutral_analysis.append("And I'm quietly catching up to the USD race leader...")
                    elif neutral.get('goal', 'usd') == 'btc' and sorted_btc_agents:
                        btc_leader_holdings = sorted_btc_agents[0].get('btc_holdings', 0)
                        neutral_holdings = neutral.get('btc_holdings', 0)
                        if neutral_holdings > btc_leader_holdings * 0.9:  # Close to leader
                            neutral_analysis.append("And I'm silently closing in on the BTC race leader...")
                    
                    discussion.append(" ".join(neutral_analysis))
                
                # Add heated debate between top performers about reaching goals first
                if oracle_agent and (sorted_usd_agents or sorted_btc_agents):
                    # Choose a top performer to debate with Oracle
                    top_performer = sorted_usd_agents[0] if sorted_usd_agents else sorted_btc_agents[0]
                    
                    debate = [
                        f"{top_performer['agent']}: I'll be the first to reach my goal with my proven strategy!",
                        
                        f"{oracle_agent['agent']}: Your strategy is flawed. My chart analysis shows {symbol} will go {prediction_direction}, "
                        f"which contradicts your position. The patterns are clear.",
                        
                        f"{top_performer['agent']}: Charts can be misleading. My {top_performer['personality']} approach considers fundamentals too.",
                        
                        f"{oracle_agent['agent']}: The numbers don't lie. RSI at {rsi:.1f}, MACD at {macd:.6f}. "
                        f"My pattern recognition algorithm is detecting a clear {pattern_text}.",
                        
                        f"{top_performer['agent']}: We'll see who reaches their goal first. That's the only metric that matters."
                    ]
                    discussion.extend(debate)
                
                # Add final competitive summary for both races
                summary = ["Race Standings:"]
                
                if sorted_usd_agents:
                    usd_summary = [
                        f"🏆 USD Race ($100 Goal):",
                        f"🥇 {sorted_usd_agents[0]['agent']}: ${sorted_usd_agents[0]['wallet_metrics']['total_value_usdt']:.2f}"
                    ]
                    
                    if len(sorted_usd_agents) > 1:
                        usd_summary.append(f"🥈 {sorted_usd_agents[1]['agent']}: ${sorted_usd_agents[1]['wallet_metrics']['total_value_usdt']:.2f}")
                    
                    if len(sorted_usd_agents) > 2:
                        usd_summary.append(f"🥉 {sorted_usd_agents[2]['agent']}: ${sorted_usd_agents[2]['wallet_metrics']['total_value_usdt']:.2f}")
                    
                    summary.append(" | ".join(usd_summary))
                
                if sorted_btc_agents:
                    btc_summary = [
                        f"🏆 BTC Race (1 BTC Goal):",
                        f"🥇 {sorted_btc_agents[0]['agent']}: {sorted_btc_agents[0].get('btc_holdings', 0):.8f} BTC"
                    ]
                    
                    if len(sorted_btc_agents) > 1:
                        btc_summary.append(f"🥈 {sorted_btc_agents[1]['agent']}: {sorted_btc_agents[1].get('btc_holdings', 0):.8f} BTC")
                    
                    if len(sorted_btc_agents) > 2:
                        btc_summary.append(f"🥉 {sorted_btc_agents[2]['agent']}: {sorted_btc_agents[2].get('btc_holdings', 0):.8f} BTC")
                    
                    summary.append(" | ".join(btc_summary))
                
                discussion.append("\n".join(summary))
                
                # Add the discussion to history with metadata
                self.discussions.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'discussion': discussion,
                    'market_data': {
                        'price': current_price,
                        'change': price_change,
                        'volume': volume,
                        'trend': trend,
                        'rsi': rsi,
                        'sma20': sma20,
                        'sma50': sma50,
                        'macd': macd,
                        'stochastic': stoch
                    },
                    'race_status': {
                        'usd_leader': sorted_usd_agents[0]['agent'] if sorted_usd_agents else None,
                        'usd_leader_value': sorted_usd_agents[0]['wallet_metrics']['total_value_usdt'] if sorted_usd_agents else 0,
                        'btc_leader': sorted_btc_agents[0]['agent'] if sorted_btc_agents else None,
                        'btc_leader_holdings': sorted_btc_agents[0].get('btc_holdings', 0) if sorted_btc_agents else 0
                    }
                })
                
                return discussion
        except Exception as e:
            print(f"Error generating discussion: {str(e)}")
            return []
            
    def analyze_market(self, symbol: str) -> List[Dict]:
        """Analyze the market for a given symbol and generate signals from all agents."""
        signals = []
        
        try:
            # Get market data
            market_data = self.get_market_data(symbol)
            if market_data.empty:
                print(f"No market data available for {symbol}")
                return []
            
            # Get current prices for all symbols
            current_prices = {}
            for sym in self.symbols:
                base_symbol = sym.split('/')[0] if '/' in sym else sym
                sym_data = self.get_market_data(sym)
                if not sym_data.empty:
                    current_prices[base_symbol] = float(sym_data['close'].iloc[-1])
            
            # Current price for the symbol being analyzed
            current_price = float(market_data['close'].iloc[-1])
            
            # Analyze with each agent
            with self.lock:
                for agent in self.agents:
                    try:
                        print(f"Agent {agent['name']} analyzing {symbol}")
                        
                        # Check if this symbol is in the agent's avoid list
                        if symbol in agent.get('avoid_pairs', []):
                            print(f"{agent['name']} avoids trading {symbol}")
                            continue
                        
                        # Adjust confidence based on preferred pairs
                        preferred_multiplier = 1.2 if symbol in agent.get('preferred_pairs', []) else 1.0
                        
                        # Analyze market data based on agent's strategy
                        analysis = self.analyze_market_data(market_data, agent['strategy'])
                        signal = self.generate_signal(analysis)
                        
                        # Boost confidence for preferred pairs
                        if preferred_multiplier > 1.0 and signal['action'] not in ['HOLD', 'WATCH']:
                            signal['confidence'] = min(0.95, signal['confidence'] * preferred_multiplier)
                            signal['reason'] += f" (Preferred pair: confidence boosted)"
                        
                        # Ensure signal has a reason field
                        if 'reason' not in signal:
                            signal['reason'] = f"Signal generated by {agent['name']} based on {agent['personality']}"
                        
                        # Get wallet metrics to check progress toward goal
                        wallet_metrics = agent['wallet'].get_performance_metrics(current_prices)
                        total_value = wallet_metrics['total_value_usdt']
                        
                        # Adjust strategy based on progress toward goal
                        if total_value < 30:  # Less than $30, be extremely aggressive
                            # Set very aggressive trading preferences
                            agent['strategy'] = {
                                'value_investing': 0.1,  # Reduce value investing weight
                                'momentum_trading': 1.0,  # Max momentum trading
                                'trend_following': 1.0,   # Max trend following
                                'swing_trading': 0.9,     # High swing trading
                                'scalping': 1.0,          # Max scalping
                                'mean_reversion': 0.8     # High mean reversion for quick profits
                            }
                            
                            # Update market beliefs to be responsive to market conditions
                            if analysis['sentiment'] == 'bullish' or analysis['sentiment'] == 'oversold':
                                agent['market_view'] = {
                                    'market_trend': 'bullish',
                                    'volatility_expectation': 'very_high',
                                    'risk_assessment': 'high_reward'
                                }
                            elif analysis['sentiment'] == 'bearish' or analysis['sentiment'] == 'overbought':
                                agent['market_view'] = {
                                    'market_trend': 'bearish',
                                    'volatility_expectation': 'very_high',
                                    'risk_assessment': 'capital_preservation'
                                }
                            else:
                                agent['market_view'] = {
                                    'market_trend': 'neutral',
                                    'volatility_expectation': 'high',
                                    'risk_assessment': 'balanced'
                                }
                            
                            # Boost confidence for all signals
                            signal['confidence'] = min(0.95, signal['confidence'] * 1.8)
                            
                            # For buy signals
                            if signal['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
                                if analysis['sentiment'] == 'bullish' or analysis['sentiment'] == 'oversold':
                                    signal['confidence'] = min(0.95, signal['confidence'] * 1.2)
                                    signal['reason'] += " (Aggressive mode: under $30, favorable buy conditions)"
                                else:
                                    # Still buy but with slightly reduced confidence if not ideal conditions
                                    signal['confidence'] = min(0.9, signal['confidence'])
                                    signal['reason'] += " (Aggressive mode: under $30, cautious buy)"
                            
                            # For sell signals
                            elif signal['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
                                if analysis['sentiment'] == 'bearish' or analysis['sentiment'] == 'overbought':
                                    signal['confidence'] = min(0.95, signal['confidence'] * 1.2)
                                    signal['reason'] += " (Aggressive mode: under $30, favorable sell conditions)"
                                else:
                                    # Still sell but with slightly reduced confidence if not ideal conditions
                                    signal['confidence'] = min(0.9, signal['confidence'])
                                    signal['reason'] += " (Aggressive mode: under $30, taking profits)"
                            
                            # Convert WATCH or HOLD to action based on market conditions
                            elif signal['action'] in ['WATCH', 'HOLD']:
                                if analysis['sentiment'] == 'bullish' or analysis['sentiment'] == 'oversold':
                                    signal['action'] = 'SCALE_IN'
                                    signal['confidence'] = 0.7
                                    signal['reason'] = "Converting to buy due to aggressive strategy (under $30, bullish conditions)"
                                elif analysis['sentiment'] == 'bearish' or analysis['sentiment'] == 'overbought':
                                    signal['action'] = 'SCALE_OUT'
                                    signal['confidence'] = 0.7
                                    signal['reason'] = "Converting to sell due to aggressive strategy (under $30, bearish conditions)"
                                else:
                                    # In neutral conditions, slightly favor buying for growth
                                    if analysis['indicators']['rsi'] < 50:
                                        signal['action'] = 'SCALE_IN'
                                        signal['confidence'] = 0.6
                                        signal['reason'] = "Converting to light buy due to aggressive strategy (under $30, neutral conditions)"
                                    else:
                                        signal['action'] = 'SCALE_OUT'
                                        signal['confidence'] = 0.6
                                        signal['reason'] = "Converting to light sell due to aggressive strategy (under $30, neutral conditions)"
                        
                        elif total_value < 40:  # Between $30-$40, be aggressive
                            # Still aggressive but slightly less
                            agent['strategy'] = {
                                'value_investing': 0.2,
                                'momentum_trading': 0.9,
                                'trend_following': 0.9,
                                'swing_trading': 0.8,
                                'scalping': 0.8
                            }
                            
                            # Boost confidence for buy signals
                            if signal['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
                                signal['confidence'] = min(0.9, signal['confidence'] * 1.5)
                                signal['reason'] += " (Aggressive mode: $30-$40)"
                        
                        elif total_value < 70:  # Between $40-$70, be moderately aggressive
                            # Moderately aggressive
                            agent['strategy'] = {
                                'value_investing': 0.3,
                                'momentum_trading': 0.8,
                                'trend_following': 0.8,
                                'swing_trading': 0.7,
                                'scalping': 0.6
                            }
                            
                            # Slightly boost confidence for buy signals
                            if signal['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
                                signal['confidence'] = min(0.85, signal['confidence'] * 1.2)
                                signal['reason'] += " (Moderately aggressive: $40-$70)"
                        
                        elif total_value > 100:  # Over $100, focus on capital preservation
                            # More conservative approach
                            agent['strategy'] = {
                                'value_investing': 0.7,
                                'momentum_trading': 0.4,
                                'trend_following': 0.5,
                                'swing_trading': 0.3,
                                'scalping': 0.2
                            }
                            
                            # Boost confidence for sell signals
                            if signal['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
                                signal['confidence'] = min(0.9, signal['confidence'] * 1.3)
                                signal['reason'] += " (Capital preservation mode: over $100)"
                        
                        # Special handling for BTC goal agents
                        if agent.get('goal', 'usd') == 'btc':
                            # Calculate BTC holdings
                            btc_holdings = 0
                            for sym, holding in wallet_metrics['holdings'].items():
                                if sym == 'BTC/USDT' or sym == 'BTC':
                                    btc_holdings += holding['amount']
                            
                            # Adjust strategy based on BTC holdings
                            if btc_holdings < 0.1:  # Less than 0.1 BTC, be extremely aggressive for BTC
                                if 'BTC' in symbol:  # Only for BTC-related pairs
                                    if signal['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
                                        signal['confidence'] = min(0.95, signal['confidence'] * 2)
                                        signal['reason'] += " (Aggressive BTC accumulation: under 0.1 BTC)"
                                    elif signal['action'] in ['WATCH', 'HOLD']:
                                        signal['action'] = 'SCALE_IN'
                                        signal['confidence'] = 0.7
                                        signal['reason'] = "Converting to buy for BTC accumulation (under 0.1 BTC)"
                            elif btc_holdings > 0.8:  # Over 0.8 BTC, focus on preserving BTC
                                if signal['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT'] and 'BTC' in symbol:
                                    signal['confidence'] = min(0.7, signal['confidence'] * 0.7)  # Reduce selling confidence
                                    signal['reason'] += " (BTC preservation mode: over 0.8 BTC)"
                        
                        # Execute trade if auto-trading is enabled
                        trade_executed = False
                        if self.auto_trading_enabled:
                            try:
                                # Make sure we're using the execute_trade method from the wallet
                                trade_executed = agent['wallet'].execute_trade(symbol, signal, current_price)
                                if trade_executed:
                                    print(f"🔄 {agent['name']} executed {signal['action']} for {symbol} at ${current_price:.2f}")
                                    print(f"   Wallet value: ${total_value:.2f} / $100.00 goal ({total_value/100*100:.1f}%)")
                            except Exception as e:
                                print(f"Error executing trade for {agent['name']}: {str(e)}")
                                # Fallback to direct buy/sell if execute_trade fails
                                if signal['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
                                    amount_usdt = agent['wallet'].balance_usdt * 0.3 * signal['confidence']
                                    # Ensure minimum trade size of $3
                                    amount_usdt = max(amount_usdt, 3.0) if agent['wallet'].balance_usdt >= 10.0 else agent['wallet'].balance_usdt * 0.3
                                    if amount_usdt >= 3.0:
                                        trade_executed = agent['wallet'].execute_buy(symbol, amount_usdt, current_price)
                                    else:
                                        print(f"Skipping fallback buy for {symbol}: amount ${amount_usdt:.2f} is below minimum $3.00")
                                elif signal['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
                                    # Handle different symbol formats
                                    base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                                    
                                    # Check if we have the asset in either format
                                    holding_symbol = None
                                    if base_symbol in agent['wallet'].holdings and agent['wallet'].holdings[base_symbol] > 0:
                                        holding_symbol = base_symbol
                                    elif symbol in agent['wallet'].holdings and agent['wallet'].holdings[symbol] > 0:
                                        holding_symbol = symbol
                                    
                                    if holding_symbol:
                                        amount_crypto = agent['wallet'].holdings[holding_symbol] * 0.5 * signal['confidence']
                                        # Calculate USDT value of the sell
                                        sell_value_usdt = amount_crypto * current_price
                                        if sell_value_usdt >= 3.0:
                                            trade_executed = agent['wallet'].execute_sell(holding_symbol, amount_crypto, current_price)
                                        else:
                                            # Try to sell more to meet minimum
                                            total_value = agent['wallet'].holdings[holding_symbol] * current_price
                                            if total_value >= 3.0:
                                                amount_crypto = 3.0 / current_price
                                                trade_executed = agent['wallet'].execute_sell(holding_symbol, amount_crypto, current_price)
                                            else:
                                                print(f"Skipping fallback sell for {holding_symbol}: value ${sell_value_usdt:.2f} is below minimum $3.00")
                        
                        # Update wallet metrics after trade
                        wallet_metrics = agent['wallet'].get_performance_metrics(current_prices)
                        personality_traits = self.get_personality_traits(agent['personality'])
                        
                        # Create signal data
                        signal_data = {
                            'agent': agent['name'],
                            'personality': personality_traits['personality'],
                            'symbol': symbol,
                            'signal': signal,
                            'risk_tolerance': agent['risk_tolerance'],
                            'strategy': personality_traits,
                            'market_view': personality_traits.get('market_beliefs', {}),
                            'wallet_metrics': wallet_metrics,
                            'trade_executed': trade_executed,
                            'timestamp': datetime.now().timestamp(),
                            'goal': agent.get('goal', 'usd'),
                            'goal_progress': f"{wallet_metrics['total_value_usdt']/100*100:.1f}%"  # Progress toward $100
                        }
                        
                        # Add to signals list
                        signals.append(signal_data)
                        
                        # Add to signals history
                        self.signals_history.append(signal_data)
                        
                        # Limit signals history size
                        if len(self.signals_history) > SYSTEM_PARAMS['max_history']:
                            self.signals_history = self.signals_history[-SYSTEM_PARAMS['max_history']:]
                        
                        print(f"Signal generated by {agent['name']} for {symbol}: {signal['action']}")
                    except Exception as e:
                        print(f"Error with agent {agent['name']} for {symbol}: {str(e)}")
                        continue
        
        except Exception as e:
            print(f"Error in analyze_market for {symbol}: {str(e)}")
        
        return signals
    
    def toggle_auto_trading(self, enabled: bool) -> None:
        """Enable or disable automatic trading."""
        self.auto_trading_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Auto-trading {status}")
        
    def run(self, interval: int = None):
        """Run the trading system continuously."""
        if interval is None:
            interval = SYSTEM_PARAMS['update_interval']
            
        print(f"Starting trading system with {len(self.agents)} agents and {len(self.symbols)} symbols")
        print(f"Update interval: {interval} seconds")
        print(f"Cache TTL: {SYSTEM_PARAMS['cache_ttl']} seconds")
        
        try:
            while True:
                start_time = time.time()
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        # Skip if we've already processed this symbol recently
                        cache_ttl = SYSTEM_PARAMS['cache_ttl']
                        if (symbol in self.last_update_time and 
                            time.time() - self.last_update_time.get(symbol, 0) < cache_ttl):
                            continue
                            
                        print(f"\nAnalyzing {symbol}...")
                        signals = self.analyze_market(symbol)
                        
                        # Update last update time for this symbol
                        self.last_update_time[symbol] = time.time()
                        
                        # Generate discussion if we have signals
                        if signals:
                            discussion = self.generate_discussion(signals)
                            if discussion:
                                print("\nTrader Discussion:")
                                for message in discussion:
                                    print(message)
                                print()
                        
                        # Record holdings after each symbol analysis
                        self.record_holdings()
                        
                        # Save state periodically
                        self._save_state()
                        
                    except Exception as e:
                        print(f"Error processing {symbol}: {str(e)}")
                        continue
                
                # Calculate time to sleep
                elapsed = time.time() - start_time
                sleep_time = max(1, interval - elapsed)
                
                print(f"\nCompleted analysis cycle in {elapsed:.1f} seconds. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nTrading system stopped by user.")
        except Exception as e:
            print(f"\nTrading system stopped due to error: {str(e)}")
        finally:
            # Save state before exiting
            self._save_state()
            print("Final state saved.")
    
    def record_holdings(self):
        """Record current holdings for each trader."""
        try:
            # Get current prices for all symbols
            current_prices = {}
            for symbol in self.symbols:
                try:
                    if symbol.endswith('/USDT'):
                        df = self.get_market_data(symbol)
                        if not df.empty:
                            base_currency = symbol.split('/')[0]
                            current_prices[base_currency] = float(df['close'].iloc[-1])
                except Exception as e:
                    print(f"Error getting price for {symbol}: {str(e)}")
            
            # Record holdings for each agent
            timestamp = datetime.now()
            for agent in self.agents:
                metrics = agent['wallet'].get_performance_metrics(current_prices)
                
                # Calculate crypto holdings value (only for non-dust amounts)
                crypto_value = metrics['total_value_usdt'] - metrics['balance_usdt']
                
                # Record holdings snapshot (filtering out dust amounts)
                holdings_snapshot = {
                    'timestamp': timestamp,
                    'total_value_usdt': metrics['total_value_usdt'],
                    'balance_usdt': metrics['balance_usdt'],
                    'crypto_value_usdt': crypto_value,
                    'holdings': {
                        symbol: {
                            'amount': amount,
                            'price_usdt': current_prices.get(symbol.split('/')[0] if '/' in symbol else symbol, 0),
                            'value_usdt': amount * current_prices.get(symbol.split('/')[0] if '/' in symbol else symbol, 0)
                        }
                        for symbol, amount in metrics['holdings'].items()
                        if amount > 1e-8  # Only include non-dust amounts
                    }
                }
                
                # Add to holdings history
                self.holdings_history[agent['name']].append(holdings_snapshot)
                
                # Keep only the last 1000 records to prevent excessive memory usage
                if len(self.holdings_history[agent['name']]) > 1000:
                    self.holdings_history[agent['name']] = self.holdings_history[agent['name']][-1000:]
                    
            print(f"Recorded holdings for {len(self.agents)} traders at {timestamp}")
        except Exception as e:
            print(f"Error recording holdings: {str(e)}")
            
    def get_holdings_history(self, agent_name: str, timeframe: str = 'all') -> List[Dict]:
        """
        Get holdings history for a specific agent.
        
        Args:
            agent_name (str): Name of the agent
            timeframe (str): Timeframe to filter (all, day, week, month)
            
        Returns:
            List[Dict]: List of holdings snapshots
        """
        if agent_name not in self.holdings_history:
            return []
            
        history = self.holdings_history[agent_name]
        
        # Filter by timeframe if needed
        if timeframe == 'day':
            cutoff = datetime.now() - timedelta(days=1)
            history = [h for h in history if h['timestamp'] >= cutoff]
        elif timeframe == 'week':
            cutoff = datetime.now() - timedelta(days=7)
            history = [h for h in history if h['timestamp'] >= cutoff]
        elif timeframe == 'month':
            cutoff = datetime.now() - timedelta(days=30)
            history = [h for h in history if h['timestamp'] >= cutoff]
            
        return history

    def get_personality_traits(self, personality_type: str) -> Dict:
        """Return personality traits based on the agent's personality type."""
        personality_traits = {
            'personality': personality_type,
            'market_beliefs': {}
        }
        
        if 'Value Investor' in personality_type:
            personality_traits['market_beliefs'] = {
                'market_trend': 'neutral',
                'risk_assessment': 'moderate',
                'time_horizon': 'long-term',
                'sentiment': 'cautious'
            }
        elif 'Tech Disruptor' in personality_type:
            personality_traits['market_beliefs'] = {
                'market_trend': 'bullish',
                'risk_assessment': 'high',
                'time_horizon': 'medium-term',
                'sentiment': 'optimistic'
            }
        elif 'Contrarian' in personality_type:
            personality_traits['market_beliefs'] = {
                'market_trend': 'bearish',
                'risk_assessment': 'very high',
                'time_horizon': 'short-term',
                'sentiment': 'pessimistic'
            }
        elif 'Macro Trader' in personality_type:
            personality_traits['market_beliefs'] = {
                'market_trend': 'neutral',
                'risk_assessment': 'moderate',
                'time_horizon': 'medium-term',
                'sentiment': 'balanced'
            }
        elif 'Swing Trader' in personality_type:
            personality_traits['market_beliefs'] = {
                'market_trend': 'volatile',
                'risk_assessment': 'high',
                'time_horizon': 'short-term',
                'sentiment': 'opportunistic'
            }
        elif 'Chart Reader' in personality_type:
            personality_traits['market_beliefs'] = {
                'market_trend': 'analytical',
                'risk_assessment': 'data-driven',
                'time_horizon': 'adaptive',
                'sentiment': 'objective'
            }
        
        return personality_traits

    def analyze_market_data(self, market_data: pd.DataFrame, strategy: Dict) -> Dict:
        """Analyze market data based on the agent's strategy."""
        analysis = {
            'price_action': {},
            'indicators': {},
            'patterns': {},
            'sentiment': 'neutral',
            'recommendation': 'HOLD'
        }
        
        if market_data.empty:
            return analysis
        
        # Extract basic price action
        current_price = market_data['close'].iloc[-1]
        prev_price = market_data['close'].iloc[-2]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        analysis['price_action'] = {
            'current_price': current_price,
            'prev_price': prev_price,
            'price_change': price_change,
            'volume': market_data['volume'].iloc[-1]
        }
        
        # Calculate technical indicators
        analysis['indicators']['sma20'] = market_data['close'].rolling(window=20).mean().iloc[-1]
        analysis['indicators']['sma50'] = market_data['close'].rolling(window=50).mean().iloc[-1]
        analysis['indicators']['rsi'] = ta.momentum.rsi(market_data['close'], window=14).iloc[-1]
        analysis['indicators']['macd'] = ta.trend.macd_diff(market_data['close']).iloc[-1]
        analysis['indicators']['stoch'] = ta.momentum.stoch(market_data['high'], market_data['low'], market_data['close']).iloc[-1]
        
        # Detect patterns based on strategy weights
        if strategy.get('pattern_recognition', 0) > 0.7:  # Oracle AI has high pattern recognition
            # Check for chart patterns
            analysis['patterns']['trend'] = 'bullish' if analysis['indicators']['sma20'] > analysis['indicators']['sma50'] else 'bearish'
            analysis['patterns']['overbought'] = analysis['indicators']['rsi'] > 70
            analysis['patterns']['oversold'] = analysis['indicators']['rsi'] < 30
            analysis['patterns']['bullish_momentum'] = analysis['indicators']['macd'] > 0
            analysis['patterns']['bearish_momentum'] = analysis['indicators']['macd'] < 0
            
            # Advanced pattern detection for Oracle AI
            price_data = market_data['close'].values
            high_data = market_data['high'].values
            low_data = market_data['low'].values
            
            # Check for double tops/bottoms (simplified)
            if len(price_data) > 20:
                recent_highs = [i for i in range(2, len(high_data)-2) if high_data[i] > high_data[i-1] and high_data[i] > high_data[i-2] and high_data[i] > high_data[i+1] and high_data[i] > high_data[i+2]]
                recent_lows = [i for i in range(2, len(low_data)-2) if low_data[i] < low_data[i-1] and low_data[i] < low_data[i-2] and low_data[i] < low_data[i+1] and low_data[i] < low_data[i+2]]
                
                if len(recent_highs) >= 2 and abs(high_data[recent_highs[-1]] - high_data[recent_highs[-2]]) / high_data[recent_highs[-2]] < 0.03:
                    analysis['patterns']['double_top'] = True
                
                if len(recent_lows) >= 2 and abs(low_data[recent_lows[-1]] - low_data[recent_lows[-2]]) / low_data[recent_lows[-2]] < 0.03:
                    analysis['patterns']['double_bottom'] = True
        
        # Determine sentiment based on indicators
        if analysis['indicators']['rsi'] > 70:
            analysis['sentiment'] = 'overbought'
        elif analysis['indicators']['rsi'] < 30:
            analysis['sentiment'] = 'oversold'
        elif analysis['indicators']['macd'] > 0 and analysis['indicators']['sma20'] > analysis['indicators']['sma50']:
            analysis['sentiment'] = 'bullish'
        elif analysis['indicators']['macd'] < 0 and analysis['indicators']['sma20'] < analysis['indicators']['sma50']:
            analysis['sentiment'] = 'bearish'
        
        # Generate recommendation
        if analysis['sentiment'] == 'bullish' or analysis['sentiment'] == 'oversold':
            analysis['recommendation'] = 'BUY'
        elif analysis['sentiment'] == 'bearish' or analysis['sentiment'] == 'overbought':
            analysis['recommendation'] = 'SELL'
        
        return analysis

    def generate_signal(self, analysis: Dict) -> Dict:
        """Generate a trading signal based on market analysis."""
        signal = {
            'action': 'HOLD',
            'confidence': 0.5,
            'reason': ''
        }
        
        # Set action based on recommendation
        if analysis['recommendation'] == 'BUY':
            # Determine strength of buy signal
            if analysis['sentiment'] == 'oversold' or (analysis['indicators']['rsi'] < 40 and analysis['indicators']['macd'] > 0):
                signal['action'] = 'STRONG_BUY'
                signal['confidence'] = 0.8
                signal['reason'] = f"Strong buy signal: RSI {analysis['indicators']['rsi']:.1f} (oversold), positive MACD"
            else:
                signal['action'] = 'BUY'
                signal['confidence'] = 0.7
                signal['reason'] = f"Buy signal: Bullish trend, RSI {analysis['indicators']['rsi']:.1f}"
        
        elif analysis['recommendation'] == 'SELL':
            # Determine strength of sell signal
            if analysis['sentiment'] == 'overbought' or (analysis['indicators']['rsi'] > 60 and analysis['indicators']['macd'] < 0):
                signal['action'] = 'STRONG_SELL'
                signal['confidence'] = 0.8
                signal['reason'] = f"Strong sell signal: RSI {analysis['indicators']['rsi']:.1f} (overbought), negative MACD"
            else:
                signal['action'] = 'SELL'
                signal['confidence'] = 0.7
                signal['reason'] = f"Sell signal: Bearish trend, RSI {analysis['indicators']['rsi']:.1f}"
        
        else:
            # Handle HOLD signals
            if analysis['indicators']['rsi'] > 45 and analysis['indicators']['rsi'] < 55:
                signal['action'] = 'HOLD'
                signal['confidence'] = 0.6
                signal['reason'] = f"Hold signal: Neutral RSI {analysis['indicators']['rsi']:.1f}"
            else:
                signal['action'] = 'WATCH'
                signal['confidence'] = 0.5
                signal['reason'] = f"Watch signal: Unclear trend, monitoring for confirmation"
        
        # Special handling for pattern recognition (Oracle AI)
        if 'patterns' in analysis and analysis['patterns']:
            pattern_reasons = []
            
            if analysis['patterns'].get('double_top', False):
                pattern_reasons.append("double top pattern detected")
                if signal['action'] in ['HOLD', 'WATCH', 'BUY']:
                    signal['action'] = 'SELL'
                    signal['confidence'] = 0.75
            
            if analysis['patterns'].get('double_bottom', False):
                pattern_reasons.append("double bottom pattern detected")
                if signal['action'] in ['HOLD', 'WATCH', 'SELL']:
                    signal['action'] = 'BUY'
                    signal['confidence'] = 0.75
            
            if pattern_reasons:
                signal['reason'] = f"Pattern-based signal: {', '.join(pattern_reasons)}"
        
        return signal

def create_dashboard(trading_system, title="AI Crypto Trading Arena", subtitle="AI Traders Battle: $20 → $100 Challenge"):
    """Create the trading dashboard."""
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(os.path.dirname(__file__), 'assets'),
        serve_locally=True,
        update_title=None
    )
    
    # Configure app to serve assets locally
    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True
    
    # Initialize signals history if not present
    if not hasattr(trading_system, 'signals_history'):
        trading_system.signals_history = []
    
    # Initialize wallets if needed
    for agent in trading_system.agents:
        if not hasattr(agent['wallet'], 'holdings'):
            agent['wallet'].holdings = {}
    
    app.layout = html.Div([
        # Header with modern design
        html.Div([
            html.Div([
                html.H1([
                    title,
                    html.Span("LIVE", className='live-badge')
                ], className='dashboard-title'),
                html.P([
                    subtitle,
                    html.Span("🤖", className='emoji'),
                    html.Span("💰", className='emoji')
                ], className='dashboard-subtitle')
            ], className='header-content'),
            html.Div([
                html.Button([
                    html.Span("Reset All Traders"),
                    html.Span("🔄", className='emoji')
                ], id='reset-traders-button', className='reset-button'),
                html.Div(id='market-summary-stats', className='market-stats')
            ], className='header-stats')
        ], className='header'),

        # Navigation with tooltips
        html.Div([
            html.Div([
                html.Button([
                    html.Span("📊", className='emoji'),
                    html.Span("Market Overview")
                ], id='nav-market-overview', className='nav-button active'),
                html.Div("View market prices and charts", className='tooltip')
            ], className='nav-item'),
            html.Div([
                html.Button([
                    html.Span("👥", className='emoji'),
                    html.Span("Traders Portfolios")
                ], id='nav-traders-comparison', className='nav-button'),
                html.Div("Compare trader performance", className='tooltip')
            ], className='nav-item')
        ], className='nav-container'),

        # Auto-refresh interval (30 seconds)
        dcc.Interval(
            id='auto-refresh-interval',
            interval=30 * 1000,  # in milliseconds
            n_intervals=0
        ),
        
        # Memory save interval (5 minutes)
        dcc.Interval(
            id='memory-save-interval',
            interval=5 * 60 * 1000,  # in milliseconds
            n_intervals=0
        ),
        
        # Refresh indicator
        html.Div([
            html.Span("Last updated: ", className="refresh-label"),
            html.Span(id="last-update-time", className="refresh-time"),
            html.Div(id="refresh-spinner", className="refresh-spinner")
        ], className="refresh-indicator"),

        # Main content area with reorganized layout
        html.Div([
            # Market Overview Tab
            html.Div([
                html.Div([
                    html.H2("Market Overview", className="section-title"),
                    html.Div([
                        html.Span("Real-time market data and charts", className="section-subtitle"),
                        html.Div(id="market-refresh-indicator", className="section-refresh-indicator")
                    ], className="section-subtitle-container")
                ], className="section-header"),
                
                # Charts Grid at the top
                html.Div(id='multi-chart-container', className='chart-grid'),
                
                # Market Overview Table
                html.Div(id='market-overview', className='market-overview'),
                
                # Trading Controls
                html.Div([
                    html.Div([
                        html.Label([
                            "Timeframe ",
                            html.Span("ℹ️", className='info-icon'),
                            html.Div("Select chart timeframe", className='tooltip')
                        ]),
                        dcc.Dropdown(
                            id='timeframe-dropdown',
                            options=[
                                {'label': '1 Hour', 'value': '1h'},
                                {'label': '4 Hours', 'value': '4h'},
                                {'label': '1 Day', 'value': '1d'}
                            ],
                            value='1h',
                            className='dropdown'
                        )
                    ], className='control-item'),
                    html.Div([
                        html.Label([
                            "Technical Indicators ",
                            html.Span("ℹ️", className='info-icon'),
                            html.Div("Choose indicators to display", className='tooltip')
                        ]),
                        dcc.Checklist(
                            id='indicator-checklist',
                            options=[
                                {'label': 'SMA', 'value': 'SMA'},
                                {'label': 'RSI', 'value': 'RSI'},
                                {'label': 'MACD', 'value': 'MACD'},
                                {'label': 'Bollinger Bands', 'value': 'BB'}
                            ],
                            value=['SMA', 'RSI'],
                            className='indicator-checklist'
                        )
                    ], className='control-item')
                ], className='trading-controls')
            ], id='market-overview-tab', style={'display': 'block'}),
            
            # Traders Portfolio Tab - Initially hidden
            html.Div([
                html.Div([
                    html.H2("Traders Portfolios", className="section-title"),
                    html.Div([
                        html.Span("Compare trader performance and strategies", className="section-subtitle"),
                        html.Div(id="traders-refresh-indicator", className="section-refresh-indicator")
                    ], className="section-subtitle-container")
                ], className="section-header"),
                
                # Performance cards with sorting
                html.Div([
                    html.Div([
                        html.H3([
                            html.Span("🏆", className="emoji"),
                            "Trader Performance"
                        ]),
                        dcc.Dropdown(
                            id='performance-sort',
                            options=[
                                {'label': 'Sort by Total Value', 'value': 'total'},
                                {'label': 'Sort by Profit %', 'value': 'profit'},
                                {'label': 'Sort by Goal Progress', 'value': 'goal'}
                            ],
                            value='total',
                            className='performance-sort'
                        )
                    ], className='performance-header'),
                    html.Div(id='traders-performance-cards', className='performance-cards-grid')
                ], className='performance-section'),
                
                # Signals Table with filter
                html.Div([
                    html.Div([
                        html.H3([
                            html.Span("📊", className="emoji"),
                            "Recent Signals"
                        ]),
                        dcc.Dropdown(
                            id='signal-filter',
                            options=[
                                {'label': 'All Signals', 'value': 'all'},
                                {'label': 'Buy Signals', 'value': 'buy'},
                                {'label': 'Strong Buy', 'value': 'strong_buy'},
                                {'label': 'Scale In', 'value': 'scale_in'},
                                {'label': 'Sell Signals', 'value': 'sell'},
                                {'label': 'Strong Sell', 'value': 'strong_sell'},
                                {'label': 'Scale Out', 'value': 'scale_out'},
                                {'label': 'Hold/Watch', 'value': 'hold'}
                            ],
                            value='all',
                            className='signal-filter'
                        )
                    ], className='signals-header'),
                    html.Div(id='signals-table', className='signals-table-container')
                ], className='signals-section'),
                
                # Agent Discussions Panel with filters
                html.Div([
                    html.Div([
                        html.H3([
                            html.Span("💬", className="emoji"),
                            "Agent Discussions"
                        ]),
                        dcc.Dropdown(
                            id='discussion-filter',
                            options=[
                                {'label': 'All Discussions', 'value': 'all'},
                                {'label': 'Bullish Views', 'value': 'bullish'},
                                {'label': 'Bearish Views', 'value': 'bearish'},
                                {'label': 'Neutral Views', 'value': 'neutral'}
                            ],
                            value='all',
                            className='discussion-filter'
                        )
                    ], className='discussions-header'),
                    html.Div(id='agent-discussions', className='discussions-container')
                ], className='discussions-section')
            ], id='traders-portfolio-tab', style={'display': 'none'}),
        ], className='main-content'),
        
        # Memory status indicator
        html.Div([
            html.Div(id='memory-status-text', className='memory-status-text'),
            html.Div(className="pulse-ring")
        ], id="memory-status", className="memory-status"),
        
        # Trade History Modal
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(id="trade-history-title", className="modal-title"),
                        html.Button("×", id="close-trade-modal", className="close-modal-btn")
                    ], className="modal-header"),
                    html.Div(id="trade-history-content", className="modal-content"),
                ], className="modal-container")
            ], className="modal-backdrop")
        ], id="trade-history-modal", style={"display": "none"})
    ], className='dashboard-container')
    
    @app.callback(
        [Output('market-overview-tab', 'style'),
         Output('traders-portfolio-tab', 'style'),
         Output('nav-market-overview', 'className'),
         Output('nav-traders-comparison', 'className')],
        [Input('nav-market-overview', 'n_clicks'),
         Input('nav-traders-comparison', 'n_clicks')]
    )
    def toggle_tabs(market_clicks, traders_clicks):
        """Toggle between market overview and traders portfolio tabs."""
        ctx = callback_context
        
        # Default to market overview tab if no clicks yet
        if not ctx.triggered:
            return {'display': 'block'}, {'display': 'none'}, 'nav-button active', 'nav-button'
        
        # Get the ID of the button that was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Switch tabs based on which button was clicked
        if button_id == 'nav-market-overview':
            print("Switching to Market Overview tab")
            return {'display': 'block'}, {'display': 'none'}, 'nav-button active', 'nav-button'
        elif button_id == 'nav-traders-comparison':
            print("Switching to Traders Portfolio tab")
            return {'display': 'none'}, {'display': 'block'}, 'nav-button', 'nav-button active'
        
        # Fallback (should not reach here)
        return {'display': 'block'}, {'display': 'none'}, 'nav-button active', 'nav-button'
    
    @app.callback(
        [Output('multi-chart-container', 'children'),
         Output('market-overview', 'children'),
         Output('signals-table', 'children')],
        [Input('auto-refresh-interval', 'n_intervals'),
         Input('timeframe-dropdown', 'value'),
         Input('indicator-checklist', 'value'),
         Input('signal-filter', 'value')]
    )
    def update_trading_view(n, timeframe, indicators, signal_filter):
        """Update the trading view components."""
        try:
            # Get market data for all symbols
            charts = []
            market_data = []
            
            for symbol in trading_system.symbols[:8]:  # Limit to 8 symbols for performance
                df = trading_system.get_market_data(symbol)
                if df.empty:
                    continue
                
                # Create figure
                fig = go.Figure()
                
                # Add candlestick chart (default style)
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol
                ))
                
                # Add indicators
                if 'SMA' in indicators:
                    sma20 = df['close'].rolling(window=20).mean()
                    sma50 = df['close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=sma20, name='SMA 20', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=df.index, y=sma50, name='SMA 50', line=dict(color='blue')))
                
                if 'RSI' in indicators:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', yaxis='y2'))
                
                # Update layout
                fig.update_layout(
                    title=symbol,
                    template='plotly_dark',
                    height=400,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                # Create chart container
                chart = html.Div(dcc.Graph(figure=fig), className='chart-container')
                charts.append(chart)
                
                # Add to market data
                current_price = float(df['close'].iloc[-1])
                prev_price = float(df['close'].iloc[-2])
                change_24h = ((current_price - prev_price) / prev_price) * 100
                
                market_data.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change_24h': change_24h,
                    'volume': float(df['volume'].iloc[-1])
                })
            
            # Create market overview
            market_overview = html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Symbol"),
                        html.Th("Price"),
                        html.Th("24h Change"),
                        html.Th("Volume")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(data['symbol']),
                            html.Td(f"${data['price']:,.2f}"),
                            html.Td(
                                f"{data['change_24h']:+.2f}%",
                                className=f"{'positive' if data['change_24h'] > 0 else 'negative'}"
                            ),
                            html.Td(f"${data['volume']:,.0f}")
                        ]) for data in market_data
                    ])
                ], className='market-overview-table')
            ])
            
            # Create signals table with filtering
            recent_signals = trading_system.signals_history[-100:]  # Get last 100 signals
            
            # Apply filter
            if signal_filter != 'all':
                if signal_filter == 'buy':
                    recent_signals = [s for s in recent_signals if any(action in s['signal']['action'].upper() for action in ['BUY', 'STRONG_BUY', 'SCALE_IN'])]
                elif signal_filter == 'strong_buy':
                    recent_signals = [s for s in recent_signals if 'STRONG_BUY' in s['signal']['action'].upper()]
                elif signal_filter == 'scale_in':
                    recent_signals = [s for s in recent_signals if 'SCALE_IN' in s['signal']['action'].upper()]
                elif signal_filter == 'sell':
                    recent_signals = [s for s in recent_signals if any(action in s['signal']['action'].upper() for action in ['SELL', 'STRONG_SELL', 'SCALE_OUT'])]
                elif signal_filter == 'strong_sell':
                    recent_signals = [s for s in recent_signals if 'STRONG_SELL' in s['signal']['action'].upper()]
                elif signal_filter == 'scale_out':
                    recent_signals = [s for s in recent_signals if 'SCALE_OUT' in s['signal']['action'].upper()]
                elif signal_filter == 'hold':
                    recent_signals = [s for s in recent_signals if any(action in s['signal']['action'].upper() for action in ['HOLD', 'WATCH'])]
            
            # Sort signals by timestamp (newest first)
            recent_signals = sorted(recent_signals, key=lambda x: x['timestamp'], reverse=True)
            
            # Create signals table
            signals_table = html.Div([
                html.Div([
                    html.Span(f"Showing {len(recent_signals)} signals", className="signals-count"),
                ], className="signals-info"),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Time"),
                        html.Th("Agent"),
                        html.Th("Symbol"),
                        html.Th("Action"),
                        html.Th("Confidence"),
                        html.Th("Status")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(datetime.fromtimestamp(signal['timestamp']).strftime('%H:%M:%S')),
                            html.Td(signal['agent']),
                            html.Td(signal['symbol']),
                            html.Td([
                                html.Div(
                                    signal['signal']['action'],
                                    className=f"signal-{signal['signal']['action'].lower()}",
                                    title=signal['signal'].get('reason', 'No reason provided')
                                ),
                                html.Div(
                                    signal['signal'].get('reason', ''),
                                    className='signal-reason',
                                    style={'display': 'none'}
                                )
                            ]),
                            html.Td(
                                html.Div([
                                    html.Div(
                                        className='confidence-bar',
                                        style={'width': f"{signal['signal']['confidence']*100}%"}
                                    ),
                                    html.Span(f"{signal['signal']['confidence']:.2f}")
                                ], className='confidence-container')
                            ),
                            html.Td(
                                html.Div([
                                    html.Span("✅" if signal.get('trade_executed', False) else "⏳"),
                                    html.Span(
                                        "Executed" if signal.get('trade_executed', False) else "Pending"
                                    )
                                ], className=f"trade-status-{'executed' if signal.get('trade_executed', False) else 'pending'}")
                            )
                        ], className='signal-row', title=signal['signal'].get('reason', 'No reason provided'))
                        for signal in recent_signals
                    ])
                ], className='signals-table-content')
            ], className='signals-table-container')
            
            return charts, market_overview, signals_table
            
        except Exception as e:
            print(f"Error updating trading view: {str(e)}")
            return [], html.Div("Error loading market data"), html.Div("Error loading signals")
    
    @app.callback(
        Output('traders-performance-cards', 'children'),
        [Input('auto-refresh-interval', 'n_intervals'),
         Input('performance-sort', 'value')]
    )
    def update_traders_comparison(n, sort_by='total'):
        """Update the traders comparison view."""
        try:
            print("Updating trader performance cards...")
            
            # Get current prices for all symbols
            current_prices = {}
            for symbol in trading_system.symbols:
                try:
                    df = trading_system.get_market_data(symbol)
                    if not df.empty:
                        # Store price both with and without /USDT suffix for compatibility
                        price = float(df['close'].iloc[-1])
                        if '/' in symbol:
                            base_currency = symbol.split('/')[0]
                            current_prices[base_currency] = price
                            current_prices[symbol] = price  # Store full symbol version too
                except Exception as e:
                    print(f"Error getting price for {symbol}: {str(e)}")
            
            # Get performance data for each agent
            performance_data = []
            for agent in trading_system.agents:
                try:
                    # Get wallet metrics
                    metrics = agent['wallet'].get_performance_metrics(current_prices)
                    
                    # Get holdings that have non-zero amounts
                    holdings_display = []
                    for symbol, holding_data in metrics.get('holdings_with_prices', {}).items():
                        if holding_data['amount'] > 1e-8:  # Only display non-dust amounts
                            holdings_display.append({
                                'symbol': symbol.split('/')[0] if '/' in symbol else symbol,
                                'amount': holding_data['amount'],
                                'price': holding_data['price'],
                                'value_usdt': holding_data['value_usdt']
                            })
                    
                    # Sort holdings by value
                    holdings_display.sort(key=lambda x: x['value_usdt'], reverse=True)
                    
                    # Get trade history and format it properly
                    trades_history = []
                    for trade in agent['wallet'].trades_history[-5:]:  # Get last 5 trades
                        if isinstance(trade, dict) and 'symbol' in trade:
                            trades_history.append({
                                'symbol': trade['symbol'],
                                'is_buy': trade.get('action', '').upper() == 'BUY',
                                'amount': trade.get('amount_crypto', 0),
                                'price': trade.get('price', 0),
                                'value': trade.get('amount_usdt', 0)
                            })
                    
                    # Calculate goal progress
                    if agent.get('goal', 'usd') == 'usd':
                        goal_progress = (metrics['total_value_usdt'] / 100.0) * 100
                        goal_status = "Goal Reached! 🏆" if goal_progress >= 100 else f"{goal_progress:.1f}% to $100"
                    else:  # BTC goal
                        btc_holdings = sum(h['amount'] for h in holdings_display if h['symbol'] == 'BTC')
                        goal_progress = btc_holdings * 100  # % of 1 BTC
                        goal_status = "Goal Reached! 🏆" if goal_progress >= 100 else f"{goal_progress:.1f}% to 1 BTC"
                    
                    performance_data.append({
                        'name': agent['name'],
                        'personality': agent['personality'],
                        'total_value': metrics['total_value_usdt'],
                        'usdt_balance': metrics['balance_usdt'],
                        'crypto_value': metrics['total_value_usdt'] - metrics['balance_usdt'],
                        'holdings': holdings_display,
                        'trades': trades_history,
                        'goal_progress': goal_progress,
                        'goal_status': goal_status,
                        'goal_type': agent.get('goal', 'usd')
                    })
                except Exception as e:
                    print(f"Error getting performance data for {agent['name']}: {str(e)}")
                    continue
            
            # Sort based on selected criteria
            if sort_by == 'total':
                performance_data.sort(key=lambda x: x['total_value'], reverse=True)
            elif sort_by == 'profit':
                performance_data.sort(key=lambda x: (x['total_value'] - 20) / 20 * 100, reverse=True)
            elif sort_by == 'goal':
                performance_data.sort(key=lambda x: x['goal_progress'], reverse=True)
            
            # Create performance cards
            cards = []
            for data in performance_data:
                card = html.Div([
                    html.Div([
                        html.H3([
                            html.Span(data['name'].replace(' AI', ''), className='agent-name'),
                            html.Span(data['personality'], className=f"agent-badge {data['personality'].lower().replace(' ', '-')}")
                        ]),
                        html.Div([
                            html.Div(f"${data['total_value']:.2f}", className='total-value'),
                            html.Div([
                                html.Span("USDT: ", className='balance-label'),
                                html.Span(f"${data['usdt_balance']:.2f}", className='balance-value')
                            ], className='balance-row'),
                            html.Div([
                                html.Span("Crypto: ", className='balance-label'),
                                html.Span(f"${data['crypto_value']:.2f}", className='balance-value')
                            ], className='balance-row')
                        ], className='balance-container'),
                        html.Div([
                            html.H4("Holdings", className='holdings-title'),
                            html.Div([
                                html.Div([
                                    html.Span(f"{h['symbol']}: ", className='holding-symbol'),
                                    html.Span(
                                        f"{h['amount']:.8f} @ ${h['price']:.2f}",
                                        className='holding-amount'
                                    ),
                                    html.Span(f"(${h['value_usdt']:.2f})", className='holding-value')
                                ], className='holding-row')
                                for h in data['holdings']
                            ], className='holdings-list') if data['holdings'] else html.Div("No crypto holdings", className='no-holdings')
                        ], className='holdings-container'),
                        # Add Trade History Section
                        html.Div([
                            html.Div([
                                html.H4("Recent Trades", className='trades-title'),
                                html.Button("View All", id={'type': 'view-trades-btn', 'index': data['name']}, className='view-all-btn')
                            ], className='trades-header'),
                            html.Div([
                                html.Div([
                                    html.Div([
                                        html.Span(
                                            "BUY" if trade['is_buy'] else "SELL",
                                            className=f"trade-type {'buy' if trade['is_buy'] else 'sell'}"
                                        ),
                                        html.Span(trade['symbol'], className='trade-symbol')
                                    ], className='trade-header'),
                                    html.Div([
                                        html.Span(
                                            f"{trade['amount']:.8f} @ ${trade['price']:.2f}",
                                            className='trade-details'
                                        ),
                                        html.Span(
                                            f"${trade['value']:.2f}",
                                            className='trade-value'
                                        )
                                    ], className='trade-info')
                                ], className='trade-row')
                                for trade in reversed(data['trades'])
                            ], className='trades-list') if data['trades'] else html.Div("No trade history", className='no-trades')
                        ], className='trades-container'),
                        html.Div([
                            html.Div(className='goal-progress-bar', children=[
                                html.Div(
                                    className='goal-progress-fill',
                                    style={'width': f"{min(100, data['goal_progress'])}%"}
                                )
                            ]),
                            html.Div([
                                html.Span(data['goal_status'], className='goal-status'),
                                html.Span(f"Goal: {'$100' if data['goal_type'] == 'usd' else '1 BTC'}", className='goal-type')
                            ], className='goal-info')
                        ], className='goal-container')
                    ], className='card-content')
                ], className='performance-card')
                cards.append(card)
            
            print(f"Generated {len(cards)} performance cards")
            return html.Div(cards, className='performance-cards-grid')
            
        except Exception as e:
            print(f"Error updating traders comparison: {str(e)}")
            return html.Div("Error loading trader data", className='error-message')
    
    @app.callback(
        Output('agent-discussions', 'children'),
        [Input('auto-refresh-interval', 'n_intervals'),
         Input('discussion-filter', 'value')]
    )
    def update_discussions(n, discussion_filter):
        """Update the agent discussions panel with filtering."""
        try:
            if not trading_system.discussions:
                return html.Div("No recent discussions", className='no-discussions')
            
            recent_discussions = trading_system.discussions[-5:]  # Get last 5 discussions
            
            # Apply filter
            if discussion_filter != 'all':
                filtered_discussions = []
                for disc in recent_discussions:
                    # Check if any message in the discussion matches the filter
                    messages = disc['discussion']
                    if discussion_filter == 'bullish' and any('BUY' in msg or 'bullish' in msg.lower() for msg in messages):
                        filtered_discussions.append(disc)
                    elif discussion_filter == 'bearish' and any('SELL' in msg or 'bearish' in msg.lower() for msg in messages):
                        filtered_discussions.append(disc)
                    elif discussion_filter == 'neutral' and any('HOLD' in msg or 'neutral' in msg.lower() for msg in messages):
                        filtered_discussions.append(disc)
                recent_discussions = filtered_discussions
            
            discussions = []
            for disc in reversed(recent_discussions):
                discussion = html.Div([
                    html.Div([
                        html.Div([
                            html.Span(disc['symbol'], className='discussion-symbol'),
                            html.Span(
                                disc['timestamp'].strftime('%H:%M:%S') if isinstance(disc['timestamp'], datetime) else 
                                datetime.fromtimestamp(disc['timestamp']).strftime('%H:%M:%S'),
                                className='discussion-time'
                            )
                        ]),
                        html.Div(className='discussion-indicator')
                    ], className='discussion-header'),
                    html.Div([
                        html.Div([
                            html.Span(
                                message.split(':')[0] if ':' in message else '',
                                className='agent-name'
                            ),
                            html.Span(
                                message.split(':', 1)[1] if ':' in message else message,
                                className='message-content'
                            )
                        ], className='discussion-message')
                        for message in disc['discussion']
                    ], className='discussion-content')
                ], className='discussion-block')
                discussions.append(discussion)
            
            return html.Div(discussions, className='discussions-list')
            
        except Exception as e:
            print(f"Error updating discussions: {str(e)}")
            return html.Div("Error loading discussions", className='error-message')
    
    @app.callback(
        Output('memory-status', 'className'),
        [Input('memory-save-interval', 'n_intervals')]
    )
    def update_memory_status(n):
        """Update the memory status indicator when state is saved."""
        if n > 0:
            trading_system._save_state()
            return "memory-status saved"
        return "memory-status"
    
    @app.callback(
        Output('market-summary-stats', 'children'),
        [Input('auto-refresh-interval', 'n_intervals')]
    )
    def update_market_summary(n):
        """Update market summary statistics."""
        try:
            stats = []
            total_volume = 0
            gainers = 0
            losers = 0
            
            for symbol in trading_system.symbols[:8]:  # Top 8 symbols
                df = trading_system.get_market_data(symbol)
                if df.empty:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                prev_price = float(df['close'].iloc[-2])
                change_24h = ((current_price - prev_price) / prev_price) * 100
                volume = float(df['volume'].iloc[-1])
                
                total_volume += volume
                if change_24h > 0:
                    gainers += 1
                else:
                    losers += 1
            
            stats = html.Div([
                html.Div([
                    html.Span("24h Volume", className='stat-label'),
                    html.Span(f"${total_volume:,.0f}", className='stat-value')
                ], className='stat-item'),
                html.Div([
                    html.Span("Gainers", className='stat-label'),
                    html.Span(f"{gainers}", className='stat-value positive')
                ], className='stat-item'),
                html.Div([
                    html.Span("Losers", className='stat-label'),
                    html.Span(f"{losers}", className='stat-value negative')
                ], className='stat-item')
            ])
            
            return stats
        except Exception as e:
            print(f"Error updating market summary: {str(e)}")
            return html.Div("Error loading market summary")
    
    @app.callback(
        Output('memory-status-text', 'children'),
        [Input('reset-traders-button', 'n_clicks')]
    )
    def reset_all_traders(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
            
        try:
            # Reset all traders
            trading_system._reset_agents()
            # Save state after reset
            trading_system._save_state()
            return "All traders reset successfully"
        except Exception as e:
            print(f"Error resetting traders: {str(e)}")
            return "Error resetting traders"
    
    @app.callback(
        [Output('last-update-time', 'children'),
         Output('refresh-spinner', 'className')],
        [Input('auto-refresh-interval', 'n_intervals')]
    )
    def update_refresh_indicator(n):
        """Update the refresh indicator with current time and show spinner during refresh."""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Show spinner for 3 seconds after refresh
        spinner_class = "refresh-spinner active" if n % 10 == 0 else "refresh-spinner"
        
        return current_time, spinner_class
    
    @app.callback(
        Output('market-refresh-indicator', 'className'),
        [Input('auto-refresh-interval', 'n_intervals')]
    )
    def update_market_refresh_indicator(n):
        """Update the market refresh indicator."""
        # Show active indicator for 3 seconds after refresh
        return "section-refresh-indicator active" if n % 10 == 0 else "section-refresh-indicator"
    
    @app.callback(
        Output('traders-refresh-indicator', 'className'),
        [Input('auto-refresh-interval', 'n_intervals')]
    )
    def update_traders_refresh_indicator(n):
        """Update the traders refresh indicator."""
        # Show active indicator for 3 seconds after refresh
        return "section-refresh-indicator active" if n % 10 == 0 else "section-refresh-indicator"
    
    @app.callback(
        [Output('trade-history-modal', 'style'),
         Output('trade-history-title', 'children'),
         Output('trade-history-content', 'children')],
        [Input({'type': 'view-trades-btn', 'index': ALL}, 'n_clicks')],
        [State({'type': 'view-trades-btn', 'index': ALL}, 'id')]
    )
    def show_trade_history(n_clicks, btn_ids):
        """Show trade history modal when a View All button is clicked."""
        ctx = callback_context
        
        if not ctx.triggered or not any(n_clicks):
            raise PreventUpdate
            
        # Find which button was clicked
        button_idx = next((i for i, n in enumerate(n_clicks) if n), None)
        if button_idx is None:
            raise PreventUpdate
            
        # Get the agent name from the button ID
        agent_name = btn_ids[button_idx]['index']
        
        # Find the agent
        agent = next((a for a in trading_system.agents if a['name'] == agent_name), None)
        if not agent:
            raise PreventUpdate
            
        # Get all trades for this agent
        trades = agent['wallet'].trades_history
        
        # Create trade history table
        if not trades:
            content = html.Div("No trade history available", className="no-trades")
        else:
            # Sort trades by timestamp (newest first)
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Create table
            content = html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Time"),
                        html.Th("Action"),
                        html.Th("Symbol"),
                        html.Th("Amount"),
                        html.Th("Price"),
                        html.Th("Value (USDT)")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            datetime.fromtimestamp(trade.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S') 
                            if isinstance(trade.get('timestamp', 0), (int, float)) 
                            else str(trade.get('timestamp', 'N/A'))
                        ),
                        html.Td(
                            trade.get('action', 'N/A'),
                            className=f"trade-action-{trade.get('action', '').lower()}"
                        ),
                        html.Td(trade.get('symbol', 'N/A')),
                        html.Td(f"{trade.get('amount_crypto', 0):.8f}"),
                        html.Td(f"${trade.get('price', 0):.2f}"),
                        html.Td(f"${trade.get('amount_usdt', 0):.2f}")
                    ], className=f"trade-history-row {'buy-row' if trade.get('action', '').upper() == 'BUY' else 'sell-row'}")
                    for trade in sorted_trades
                ])
            ], className="trade-history-table")
        
        return {'display': 'block'}, f"Trade History - {agent_name}", content
    
    @app.callback(
        Output('trade-history-modal', 'style', allow_duplicate=True),
        [Input('close-trade-modal', 'n_clicks')],
        prevent_initial_call=True
    )
    def close_trade_history(n_clicks):
        """Close the trade history modal."""
        if n_clicks:
            return {'display': 'none'}
        raise PreventUpdate
    
    return app

def run_dashboard(trading_system):
    """Run the spot trading dashboard."""
    app = create_dashboard(trading_system)
    app.run_server(debug=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    # Create trading system instance
    trading_system = TradingSystem()
    
    # Start the trading system in a background thread
    trading_thread = Thread(target=trading_system.run)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Create and run the dashboard
<<<<<<< HEAD
    app = create_dashboard(trading_system)
    app.run_server(host='0.0.0.0', port=8050, debug=False) 
=======
    run_dashboard(trading_system) 
>>>>>>> 59355cecc9b582561f2aaf98fa89d36d9d48ea41
