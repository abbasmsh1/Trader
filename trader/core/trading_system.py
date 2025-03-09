import os
import time
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock, Thread

from trader.data.market_data import MarketDataFetcher
from trader.agents.base_agent import BaseAgent

class TradingSystem:
    """
    Core trading system that manages agents, market data, and trading operations.
    """
    def __init__(self, symbols: List[str] = None, agents: List[BaseAgent] = None, goal_filter: str = None):
        """
        Initialize the trading system.
        
        Args:
            symbols: List of trading pair symbols
            agents: List of trading agents
            goal_filter: Filter agents by goal type ('usd' or 'btc')
        """
        # Default symbols
        default_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
            'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT', 'XRP/USDT'
        ]
        
        self.symbols = symbols or default_symbols
        self.agents = agents or []
        self.goal_filter = goal_filter
        
        # Initialize market data fetcher
        self.data_fetcher = MarketDataFetcher()
        
        # System parameters
        self.update_interval = 60  # seconds
        self.save_interval = 300   # seconds
        self.max_history = 1000    # maximum number of signals to keep
        self.min_trade_amount = 3.0  # minimum trade amount in USDT
        
        # State variables
        self.running = False
        self.last_update = datetime.now()
        self.last_save = datetime.now()
        self.signals_history = []
        self.discussions_history = []
        self.lock = Lock()  # Thread safety
        
    def add_agent(self, agent: BaseAgent) -> None:
        """
        Add a trading agent to the system.
        
        Args:
            agent: Trading agent to add
        """
        self.agents.append(agent)
        
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with market data
        """
        return self.data_fetcher.get_market_data(symbol, timeframe, limit=limit)
        
    def analyze_markets(self) -> List[Dict]:
        """
        Analyze all markets with all agents.
        
        Returns:
            List of signals generated
        """
        signals = []
        
        for symbol in self.symbols:
            for agent in self.agents:
                try:
                    # Get market data for the agent's timeframe
                    data = self.get_market_data(symbol, agent.timeframe)
                    
                    if data.empty:
                        continue
                        
                    # Analyze market data
                    analysis = agent.analyze_market(symbol, data)
                    
                    # Generate signal if confidence is provided
                    if 'confidence' in analysis:
                        confidence = analysis['confidence']
                        reason = analysis.get('reason', '')
                        
                        signal = agent.generate_signal(symbol, confidence, reason)
                        signals.append(signal)
                        
                        # Add to system history
                        with self.lock:
                            self.signals_history.append({
                                'timestamp': signal['timestamp'],
                                'agent': agent.name,
                                'symbol': symbol,
                                'action': signal['action'],
                                'confidence': confidence,
                                'reason': reason
                            })
                            
                            # Trim history if needed
                            if len(self.signals_history) > self.max_history:
                                self.signals_history = self.signals_history[-self.max_history:]
                                
                except Exception as e:
                    print(f"Error analyzing {symbol} with {agent.name}: {e}")
                    
        return signals
        
    def execute_signals(self, signals: List[Dict]) -> None:
        """
        Execute trading signals.
        
        Args:
            signals: List of signals to execute
        """
        executed_signals = []
        
        for signal in signals:
            try:
                agent_name = signal.get('agent_name', '')
                symbol = signal.get('symbol', '')
                action = signal.get('action', '')
                
                # Find the agent
                agent = next((a for a in self.agents if a.name == agent_name), None)
                
                if not agent:
                    continue
                    
                # Skip WATCH and HOLD signals for execution
                if action in ['WATCH', 'HOLD']:
                    # Just log the signal
                    print(f"{agent.name} decided to {action} {symbol}")
                    continue
                
                # Execute the signal
                success = agent.execute_signal(signal)
                
                if success:
                    executed_signals.append(signal)
                    print(f"{agent.name} executed {action} for {symbol}")
                    
                    # If this was a buy signal, check if we should set a stop loss
                    if action in ['BUY', 'STRONG_BUY', 'SCALE_IN'] and hasattr(agent.wallet, 'set_stop_loss'):
                        # Set a stop loss based on the agent's risk tolerance
                        stop_loss_pct = 0.05 * (1 - agent.risk_tolerance)  # Higher risk tolerance = tighter stop loss
                        agent.wallet.set_stop_loss(symbol, signal.get('price', 0), stop_loss_pct)
                        
            except Exception as e:
                print(f"Error executing signal: {e}")
                
        # Update stop losses for all agents
        self._update_stop_losses()
        
        return executed_signals
        
    def _update_stop_losses(self) -> None:
        """Update stop losses for all agents."""
        # Get current prices for all symbols
        symbols = self.symbols
        prices = self.data_fetcher.fetch_multiple_prices(symbols)
        
        for agent in self.agents:
            if hasattr(agent.wallet, 'update_stop_losses'):
                executed_stop_losses = agent.wallet.update_stop_losses(prices)
                
                # Log executed stop losses
                for trade in executed_stop_losses:
                    print(f"{agent.name} executed STOP_LOSS for {trade['symbol']} at ${trade['price']:.2f}")
        
    def generate_discussion(self) -> Dict:
        """
        Generate a discussion among agents about the market.
        
        Returns:
            Discussion dictionary
        """
        # Select a random symbol to discuss
        if not self.symbols:
            return {}
            
        symbol = np.random.choice(self.symbols)
        
        # Get current price
        current_price = self.data_fetcher.fetch_current_price(symbol)
        
        # Get market data
        data = self.get_market_data(symbol, '1h')
        
        if data.empty:
            return {}
            
        # Get agents' analyses
        analyses = []
        
        for agent in self.agents:
            try:
                analysis = agent.analyze_market(symbol, data)
                
                if 'confidence' in analysis:
                    analyses.append({
                        'agent': agent.name,
                        'personality': agent.personality,
                        'confidence': analysis['confidence'],
                        'reason': analysis.get('reason', ''),
                        'market_beliefs': agent.market_beliefs
                    })
            except Exception as e:
                print(f"Error getting analysis from {agent.name}: {e}")
                
        if not analyses:
            return {}
            
        # Categorize agents by sentiment
        bulls = [a for a in analyses if a['confidence'] > 0.3]
        bears = [a for a in analyses if a['confidence'] < -0.3]
        neutrals = [a for a in analyses if -0.3 <= a['confidence'] <= 0.3]
        
        # Sort by confidence
        bulls.sort(key=lambda x: x['confidence'], reverse=True)
        bears.sort(key=lambda x: x['confidence'])
        
        # Generate discussion
        messages = []
        
        # Market overview
        messages.append({
            'agent': 'Market Overview',
            'message': f"Current price of {symbol}: ${current_price:.2f}"
        })
        
        # Bull arguments
        if bulls:
            bull = bulls[0]
            messages.append({
                'agent': bull['agent'],
                'message': f"I'm bullish on {symbol}. {bull['reason']}"
            })
            
        # Bear arguments
        if bears:
            bear = bears[0]
            messages.append({
                'agent': bear['agent'],
                'message': f"I'm bearish on {symbol}. {bear['reason']}"
            })
            
        # Neutral perspective
        if neutrals:
            neutral = neutrals[0]
            messages.append({
                'agent': neutral['agent'],
                'message': f"I'm neutral on {symbol}. {neutral['reason']}"
            })
            
        # Create discussion
        discussion = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'price': current_price,
            'messages': messages
        }
        
        # Add to history
        with self.lock:
            self.discussions_history.append(discussion)
            
            # Trim history if needed
            if len(self.discussions_history) > self.max_history:
                self.discussions_history = self.discussions_history[-self.max_history:]
                
        return discussion
        
    def update(self) -> None:
        """
        Update the trading system.
        """
        with self.lock:
            print(f"Updating trading system at {datetime.now()}")
            
            # Analyze markets and generate signals
            signals = self.analyze_markets()
            
            # Execute signals
            executed_signals = self.execute_signals(signals)
            
            # Generate discussion
            discussion = self.generate_discussion()
            
            # Update last update time
            self.last_update = datetime.now()
            
            # Save state if needed
            if (datetime.now() - self.last_save).total_seconds() >= self.save_interval:
                self._save_state()
                self.last_save = datetime.now()
        
    def run(self) -> None:
        """Run the trading system in a loop."""
        self.running = True
        
        # Load previous state if available
        self._load_state()
        
        print(f"Starting trading system with {len(self.agents)} agents and {len(self.symbols)} symbols")
        print(f"Update interval: {self.update_interval} seconds")
        
        try:
            while self.running:
                self.update()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("Trading system stopped by user")
            
        except Exception as e:
            print(f"Error in trading system: {e}")
            
        finally:
            self._save_state()
            self.running = False
            
    def stop(self) -> None:
        """Stop the trading system."""
        self.running = False
        
    def _save_state(self) -> None:
        """Save the system state to disk."""
        try:
            state = {
                'signals_history': self.signals_history,
                'discussions_history': self.discussions_history,
                'agents': [
                    {
                        'name': agent.name,
                        'wallet': agent.wallet,
                        'signals_history': agent.signals_history
                    }
                    for agent in self.agents
                ]
            }
            
            with open('data/trading_state.pkl', 'wb') as f:
                pickle.dump(state, f)
                
            print(f"Saved trading system state at {datetime.now()}")
            
        except Exception as e:
            print(f"Error saving state: {e}")
            
    def _load_state(self) -> None:
        """Load the system state from disk."""
        try:
            if os.path.exists('data/trading_state.pkl'):
                with open('data/trading_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                    
                self.signals_history = state.get('signals_history', [])
                self.discussions_history = state.get('discussions_history', [])
                
                # Load agent states
                for agent_state in state.get('agents', []):
                    agent_name = agent_state.get('name', '')
                    
                    # Find the agent
                    agent = next((a for a in self.agents if a.name == agent_name), None)
                    
                    if agent:
                        agent.wallet = agent_state.get('wallet', agent.wallet)
                        agent.signals_history = agent_state.get('signals_history', [])
                        
                print(f"Loaded saved state with {len(self.signals_history)} signals and {len(state.get('agents', []))} agent wallets")
            else:
                # No saved state, make initial BTC purchases for agents with BTC goal
                if self.goal_filter == 'btc':
                    self._make_initial_btc_purchases()
                
        except Exception as e:
            print(f"Error loading state: {e}")
            
    def _make_initial_btc_purchases(self) -> None:
        """Make initial BTC purchases for agents with BTC goal."""
        try:
            # Get current BTC price
            btc_price = self.data_fetcher.fetch_current_price("BTC/USDT")
            
            if btc_price is None or btc_price <= 0:
                print("Error: Could not fetch BTC price for initial purchase")
                return
                
            print(f"Current BTC price: ${btc_price:.2f}")
            
            # Make initial BTC purchase for each agent
            for agent in self.agents:
                # Use 80% of initial balance to buy BTC
                initial_balance = agent.wallet.balance_usdt
                amount_usdt = initial_balance * 0.8
                
                # Create a buy signal
                signal = {
                    'timestamp': datetime.now(),
                    'agent_name': agent.name,
                    'symbol': "BTC/USDT",
                    'action': 'BUY',
                    'confidence': 0.8,
                    'amount_usdt': amount_usdt,
                    'amount_crypto': amount_usdt / btc_price if btc_price > 0 else 0,
                    'price': btc_price,
                    'reason': "Initial BTC purchase for BTC goal"
                }
                
                # Execute the signal
                success = agent.execute_signal(signal)
                
                if success:
                    print(f"Made initial BTC purchase for {agent.name}: ${amount_usdt:.2f} at ${btc_price:.2f}")
                    self.signals_history.append(signal)
                else:
                    print(f"Failed to make initial BTC purchase for {agent.name}")
                    
        except Exception as e:
            print(f"Error making initial BTC purchases: {e}")
            
    def get_agent_performance(self) -> List[Dict]:
        """
        Get performance metrics for all agents.
        
        Returns:
            List of performance metrics dictionaries
        """
        performance = []
        
        for agent in self.agents:
            try:
                metrics = agent.get_performance_metrics()
                performance.append(metrics)
            except Exception as e:
                print(f"Error getting performance for {agent.name}: {e}")
                
        # Sort by total value
        performance.sort(key=lambda x: x.get('total_value_usdt', 0), reverse=True)
        
        return performance
        
    def reset_all_agents(self) -> None:
        """Reset all agents to their initial state."""
        with self.lock:
            for agent in self.agents:
                agent.reset()
                
            self.signals_history = []
            self.discussions_history = []
            
            print("All agents have been reset to their initial state") 