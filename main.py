import os
from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta
import time
from data.market_data import MarketDataFetcher
from agents.value_investor import ValueInvestor
from agents.tech_disruptor import TechDisruptor
from agents.trend_follower import TrendFollower
from agents.contrarian_trader import ContrarianTrader
from agents.macro_trader import MacroTrader
from agents.swing_trader import SwingTrader
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from threading import Thread, Lock
import plotly.io as pio
from config.settings import DEFAULT_SYMBOLS, SYSTEM_PARAMS, TIMEFRAMES
import ta
from plotly.subplots import make_subplots
from agents.wallet import Wallet

# Set default plotly theme
pio.templates.default = "plotly_dark"

class TradingSystem:
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the trading system with personality-based agents.
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
        self.agents = [
            ValueInvestor(name="Warren Buffett AI", timeframe='1d'),
            TechDisruptor(name="Elon Musk AI", timeframe='1h'),
            ContrarianTrader(name="Michael Burry AI", timeframe='1d'),
            MacroTrader(name="Ray Dalio AI", timeframe='1d'),
            SwingTrader(name="Jesse Livermore AI", timeframe='4h'),
            TrendFollower(name="Paul Tudor Jones AI", risk_tolerance=0.8, timeframe='4h')
        ]
        
        # Add and configure high-risk trader
        high_risk_trader = TrendFollower(name="High-Risk Trader", risk_tolerance=1.0, timeframe='1h')
        high_risk_trader.set_strategy_preferences({
            'value_investing': 0.1,
            'momentum_trading': 1.0,
            'trend_following': 1.0,
            'swing_trading': 0.9,
            'scalping': 1.0
        })
        high_risk_trader.update_market_beliefs({
            'market_trend': 'extremely_bullish',
            'volatility_expectation': 'very_high',
            'risk_assessment': 'high_reward'
        })
        self.agents.append(high_risk_trader)
        
        self.signals_history = []
        self.market_data_cache = {}
        self.last_update = {}
        self.auto_trading_enabled = True
        
        # Holdings history for each trader
        self.holdings_history = {agent.name: [] for agent in self.agents}
        
        # Thread safety
        self.lock = Lock()
        
        # Reset all agents' wallets to initial state
        self._reset_agents()
        
        # Make initial BTC purchase for all agents
        self._make_initial_btc_purchase()
        
        self.discussions = []  # Store agent discussions
        
    def _reset_agents(self):
        """Reset all agents to their initial state."""
        print("Resetting all agents to initial state...")
        for agent in self.agents:
            agent.wallet = Wallet(initial_balance_usdt=20.0)
            agent.wallet.holdings = {}
            agent.wallet.trades_history = []
            print(f"Reset {agent.name}'s wallet to $20.0 USDT")
            
        # Clear signals history
        self.signals_history = []
        
        # Clear holdings history
        self.holdings_history = {agent.name: [] for agent in self.agents}
        
        # Clear discussions
        self.discussions = []
        
        # Save the reset state
        self._save_state()
        print("Trading system reset complete!")
    
    def _load_state(self):
        """Load saved state from disk if available."""
        # Force reset on startup
        self._reset_agents()
        self._state_loaded = True
        print("Forced reset of trading system on startup")
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
                        self.agents[i].wallet.balance_usdt = agent_state.get('balance_usdt', 20.0)
                        self.agents[i].wallet.holdings = agent_state.get('holdings', {})
                        self.agents[i].wallet.trades_history = agent_state.get('trades_history', [])
                
                # Restore holdings history
                self.holdings_history = state.get('holdings_history', {agent.name: [] for agent in self.agents})
                
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
                        'name': agent.name,
                        'balance_usdt': agent.wallet.balance_usdt,
                        'holdings': agent.wallet.holdings,
                        'trades_history': agent.wallet.trades_history
                    }
                    for agent in self.agents
                ],
                'holdings_history': self.holdings_history,
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
        """Make an initial purchase of BTC for all agents."""
        try:
            # Get current BTC price
            btc_data = self.get_market_data("BTC/USDT")
            if btc_data.empty:
                print("Could not get BTC price for initial purchase")
                return
                
            btc_price = float(btc_data['close'].iloc[-1])
            print(f"Making initial BTC purchase at ${btc_price:.2f}")
            
            # Each agent buys BTC with 80% of their initial balance
            for agent in self.agents:
                initial_usdt = agent.wallet.balance_usdt
                purchase_amount = initial_usdt * 0.8  # Use 80% of initial balance
                
                success = agent.wallet.execute_buy("BTC/USDT", purchase_amount, btc_price)
                if success:
                    btc_amount = purchase_amount / btc_price
                    print(f"🔄 {agent.name} made initial BTC purchase: {btc_amount:.8f} BTC (${purchase_amount:.2f})")
                    
                    # Get personality traits
                    personality_traits = agent.get_personality_traits()
                    
                    # Add to signals history
                    self.signals_history.append({
                        'agent': agent.name,
                        'personality': personality_traits['personality'],
                        'symbol': "BTC/USDT",
                        'signal': {
                            'action': 'STRONG_BUY', 
                            'confidence': 0.8, 
                            'reason': 'Initial purchase to grow $20 to $100'
                        },
                        'risk_tolerance': agent.risk_tolerance,
                        'strategy': personality_traits,
                        'market_view': personality_traits.get('market_beliefs', {}),
                        'wallet_metrics': agent.wallet.get_performance_metrics({'BTC/USDT': btc_price}),
                        'trade_executed': True,
                        'timestamp': datetime.now().timestamp()
                    })
                else:
                    print(f"❌ {agent.name} failed to make initial BTC purchase")
                    
            # Set more aggressive trading strategies for all agents
            self._set_aggressive_strategies()
                
        except Exception as e:
            print(f"Error making initial BTC purchase: {str(e)}")
            
    def _set_aggressive_strategies(self):
        """Set more aggressive trading strategies for all agents to reach $100 goal faster."""
        for agent in self.agents:
            # Update strategy preferences to be more aggressive
            agent.set_strategy_preferences({
                'value_investing': 0.3,
                'momentum_trading': 0.8,
                'trend_following': 0.9,
                'swing_trading': 0.7,
                'scalping': 0.6
            })
            
            # Update market beliefs to be more optimistic
            agent.update_market_beliefs({
                'market_trend': 'strongly_bullish',
                'volatility_expectation': 'high',
                'risk_assessment': 'opportunity'
            })
            
            print(f"Set aggressive trading strategy for {agent.name}")
        
    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data with caching."""
        try:
            current_time = time.time()
            cache_ttl = SYSTEM_PARAMS.get('cache_ttl', 60)  # 1 minute default TTL
            
            with self.lock:
                # Check if we have cached data that's still fresh
                if (symbol in self.market_data_cache and 
                    symbol in self.last_update and 
                    current_time - self.last_update[symbol] < cache_ttl):
                    print(f"Using cached data for {symbol}")
                    return self.market_data_cache[symbol]
                
                # If not, fetch new data
                print(f"Fetching new data for {symbol}")
                
                # For coin-to-coin pairs, we need to calculate the ratio
                if '/' in symbol and not symbol.endswith('/USDT'):
                    base, quote = symbol.split('/')
                    
                    # Get data for both coins in USDT
                    base_data = self.data_fetcher.fetch_market_data(f"{base}/USDT")
                    quote_data = self.data_fetcher.fetch_market_data(f"{quote}/USDT")
                    
                    if base_data.empty or quote_data.empty:
                        print(f"Could not fetch data for {symbol}")
                        return pd.DataFrame()
                    
                    # Ensure both dataframes have the same timestamps
                    common_index = base_data.index.intersection(quote_data.index)
                    base_data = base_data.loc[common_index]
                    quote_data = quote_data.loc[common_index]
                    
                    # Calculate the ratio for OHLCV
                    df = pd.DataFrame(index=common_index)
                    df['open'] = base_data['open'] / quote_data['open']
                    df['high'] = base_data['high'] / quote_data['low']  # Max ratio possible
                    df['low'] = base_data['low'] / quote_data['high']   # Min ratio possible
                    df['close'] = base_data['close'] / quote_data['close']
                    df['volume'] = base_data['volume'] * base_data['close']  # Volume in base currency value
                    
                    print(f"Successfully calculated {symbol} ratio: {len(df)} rows")
                else:
                    # Regular USDT pair
                    df = self.data_fetcher.fetch_market_data(symbol)
                    print(f"Successfully fetched data for {symbol}: {len(df)} rows")
                
                if not df.empty:
                    self.market_data_cache[symbol] = df
                    self.last_update[symbol] = current_time
                
                return df
        except Exception as e:
            print(f"Error in get_market_data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def generate_discussion(self, signals: List[Dict]) -> str:
        """Generate a discussion between agents about their trading signals."""
        if not signals:
            return ""
            
        try:
            # Group signals by action type
            bullish_agents = [s for s in signals if s['signal']['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']]
            bearish_agents = [s for s in signals if s['signal']['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT']]
            neutral_agents = [s for s in signals if s['signal']['action'] in ['HOLD', 'WATCH']]
            
            symbol = signals[0]['symbol']
            discussion = []
            
            # Start with a bullish perspective if any
            if bullish_agents:
                bull = bullish_agents[0]
                discussion.append(f"{bull['agent']}: I'm seeing a strong opportunity in {symbol}. "
                               f"My {bull['personality']} approach suggests {bull['signal']['action']} "
                               f"with {bull['signal']['confidence']:.0%} confidence. {bull['signal'].get('reason', '')}")
                
                # Add supporting or opposing views
                if len(bullish_agents) > 1:
                    supporter = bullish_agents[1]
                    discussion.append(f"{supporter['agent']}: I agree! {supporter['signal'].get('reason', 'The technical indicators are aligning.')}")
                
            # Add bearish perspective
            if bearish_agents:
                bear = bearish_agents[0]
                discussion.append(f"{bear['agent']}: I disagree. {bear['signal'].get('reason', 'The risks are too high.')} "
                               f"I'm {bear['signal']['action']} with {bear['signal']['confidence']:.0%} confidence.")
                
            # Add neutral perspective
            if neutral_agents:
                neutral = neutral_agents[0]
                discussion.append(f"{neutral['agent']}: Let's not be hasty. {neutral['signal'].get('reason', 'The market needs more time to show its direction.')}")
                
            # Add a concluding remark from a high-performing agent
            top_agent = max(signals, key=lambda x: x.get('wallet_metrics', {}).get('total_value_usdt', 0))
            discussion.append(f"{top_agent['agent']}: Based on my performance so far (${top_agent['wallet_metrics']['total_value_usdt']:.2f}), "
                           f"I'm sticking to my {top_agent['signal']['action']} position.")
            
            # Add the discussion to history
            self.discussions.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'discussion': discussion
            })
            
            return discussion
        except Exception as e:
            print(f"Error generating discussion: {str(e)}")
            return []
            
    def analyze_market(self, symbol: str) -> List[Dict]:
        """Analyze market data using all agents with a focus on reaching $100 goal."""
        try:
            print(f"Starting market analysis for {symbol}")
            market_data = self.get_market_data(symbol)
            if market_data.empty:
                print(f"No market data available for {symbol}")
                return []
            
            market_data.name = symbol
            current_price = float(market_data['close'].iloc[-1])
            
            # Create a dictionary of current prices for all crypto assets
            # For coin-to-coin pairs, we need both coins
            if '/' in symbol:
                base_currency, quote_currency = symbol.split('/')
                
                # For USDT pairs, the quote is USDT
                if quote_currency == 'USDT':
                    current_prices = {base_currency: current_price}
                else:
                    # For coin-to-coin pairs, we need to get both prices in USDT
                    base_usdt_data = self.get_market_data(f"{base_currency}/USDT")
                    quote_usdt_data = self.get_market_data(f"{quote_currency}/USDT")
                    
                    if not base_usdt_data.empty and not quote_usdt_data.empty:
                        base_price = float(base_usdt_data['close'].iloc[-1])
                        quote_price = float(quote_usdt_data['close'].iloc[-1])
                        
                        current_prices = {
                            base_currency: base_price,
                            quote_currency: quote_price
                        }
                    else:
                        current_prices = {}
            else:
                # Fallback for any other format
                current_prices = {}
            
            # Add prices for other symbols in the portfolio
            for other_symbol in self.symbols:
                if other_symbol != symbol and other_symbol.endswith('/USDT'):
                    other_data = self.get_market_data(other_symbol)
                    if not other_data.empty:
                        other_base = other_symbol.split('/')[0]
                        other_price = float(other_data['close'].iloc[-1])
                        current_prices[other_base] = other_price
            
            print(f"Current price for {symbol}: {current_price}")
            
            signals = []
            for agent in self.agents:
                try:
                    print(f"Agent {agent.name} analyzing {symbol}")
                    analysis = agent.analyze_market(market_data)
                    signal = agent.generate_signal(analysis)
                    
                    # Ensure signal has a reason field
                    if 'reason' not in signal:
                        signal['reason'] = f"Signal generated by {agent.name} based on {agent.personality}"
                    
                    # Get wallet metrics to check progress toward $100 goal
                    wallet_metrics = agent.wallet.get_performance_metrics(current_prices)
                    total_value = wallet_metrics['total_value_usdt']
                    
                    # Adjust strategy based on progress toward goal
                    if total_value < 40:  # Less than $40, be very aggressive
                        # Increase confidence for buy signals
                        if signal['action'] in ['BUY', 'STRONG_BUY', 'SCALE_IN']:
                            signal['confidence'] = min(1.0, signal['confidence'] * 1.5)
                            signal['reason'] += " (Boosted: Aggressive growth strategy to reach $100)"
                    elif total_value < 70:  # Between $40 and $70, moderately aggressive
                        if signal['action'] in ['BUY', 'STRONG_BUY']:
                            signal['confidence'] = min(1.0, signal['confidence'] * 1.2)
                            signal['reason'] += " (Boosted: Pushing toward $100 goal)"
                    elif total_value >= 100:  # Goal reached, focus on preservation
                        if signal['action'] in ['SELL', 'STRONG_SELL', 'SCALE_OUT']:
                            signal['confidence'] = min(1.0, signal['confidence'] * 1.3)
                            signal['reason'] += " (Boosted: Securing profits after reaching $100 goal)"
                    
                    # Execute trade if auto-trading is enabled
                    trade_executed = False
                    if self.auto_trading_enabled:
                        trade_executed = agent.execute_trade(symbol, signal, current_price)
                        if trade_executed:
                            print(f"🔄 {agent.name} executed {signal['action']} for {symbol} at ${current_price:.2f}")
                            print(f"   Wallet value: ${total_value:.2f} / $100.00 goal ({total_value/100*100:.1f}%)")
                            
                            # Save state after each trade
                            self._save_state()
                    
                    # Update wallet metrics after trade
                    wallet_metrics = agent.wallet.get_performance_metrics(current_prices)
                    personality_traits = agent.get_personality_traits()
                    
                    signals.append({
                        'agent': agent.name,
                        'personality': personality_traits['personality'],
                        'symbol': symbol,
                        'signal': signal,
                        'risk_tolerance': agent.risk_tolerance,
                        'strategy': personality_traits,
                        'market_view': personality_traits.get('market_beliefs', {}),
                        'wallet_metrics': wallet_metrics,
                        'trade_executed': trade_executed,
                        'timestamp': datetime.now().timestamp(),
                        'goal_progress': f"{wallet_metrics['total_value_usdt']/100*100:.1f}%"  # Progress toward $100
                    })
                    print(f"Signal generated by {agent.name} for {symbol}: {signal['action']}")
                except Exception as e:
                    print(f"Error with agent {agent.name} for {symbol}: {str(e)}")
                    continue
            
            # Generate discussion about the signals
            if signals:
                discussion = self.generate_discussion(signals)
                print("\nAgent Discussion:")
                for message in discussion:
                    print(message)
                print()
            
            return signals
        except Exception as e:
            print(f"Error in analyze_market for {symbol}: {str(e)}")
            return []
    
    def toggle_auto_trading(self, enabled: bool) -> None:
        """Enable or disable automatic trading."""
        self.auto_trading_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Auto-trading {status}")
        
    def run(self, interval: int = SYSTEM_PARAMS['update_interval']):
        """
        Run the trading system continuously.
        
        Args:
            interval (int): Update interval in seconds
        """
        print("Starting trading system...")
        last_save_time = time.time()
        last_record_time = time.time()
        save_interval = 300  # Save state every 5 minutes
        record_interval = 60  # Record holdings every 1 minute
        
        while True:
            try:
                for symbol in self.symbols:
                    print(f"\nProcessing {symbol}...")
                    signals = self.analyze_market(symbol)
                    if signals:
                        # Add to signals history
                        self.signals_history.extend(signals)
                        # Keep only the most recent 1000 signals
                        if len(self.signals_history) > 1000:
                            self.signals_history = self.signals_history[-1000:]
                
                # Record holdings periodically
                current_time = time.time()
                if current_time - last_record_time > record_interval:
                    self.record_holdings()
                    last_record_time = current_time
                    print("Recorded trader holdings")
                
                # Save state periodically
                if current_time - last_save_time > save_interval:
                    self._save_state()
                    last_save_time = current_time
                    print("Saved trading system state")
                
                print(f"Sleeping for {interval} seconds...")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nTrading system stopped by user")
                self._save_state()  # Save state before exiting
                break
            except Exception as e:
                print(f"Error in trading system run loop: {str(e)}")
                time.sleep(10)  # Wait a bit before retrying
    
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
                metrics = agent.wallet.get_performance_metrics(current_prices)
                
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
                self.holdings_history[agent.name].append(holdings_snapshot)
                
                # Keep only the last 1000 records to prevent excessive memory usage
                if len(self.holdings_history[agent.name]) > 1000:
                    self.holdings_history[agent.name] = self.holdings_history[agent.name][-1000:]
                    
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

def create_dashboard(trading_system):
    """Create the trading dashboard."""
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
        assets_folder='assets'  # Point to assets folder for static files
    )
    
    # Initialize signals history if not present
    if not hasattr(trading_system, 'signals_history'):
        trading_system.signals_history = []
    
    # Initialize wallets if needed
    for agent in trading_system.agents:
        if not hasattr(agent, 'wallet'):
            agent.wallet = Wallet(initial_balance_usdt=20.0)
        if not hasattr(agent.wallet, 'holdings'):
            agent.wallet.holdings = {}
    
    app.layout = html.Div([
        # Header with new style
        html.Div([
            html.Div([
                html.H1([
                    "AI Crypto Trading Arena ",
                    html.Span("LIVE", className='live-badge')
                ], className='dashboard-title'),
                html.P([
                    "AI Traders Battle: $20 → $100 Challenge ",
                    html.Span("🤖", className='emoji'),
                    html.Span("💰", className='emoji')
                ], className='dashboard-subtitle')
            ], className='header-content'),
            html.Div([
                html.Button("Reset All Traders", id='reset-traders-button', className='reset-button'),
                html.Div(id='market-summary-stats', className='market-stats')
            ], className='header-stats')
        ], className='header'),

        # Navigation with tooltips
        html.Div([
            html.Div([
                html.Button("Market Overview", id='nav-market-overview', className='nav-button active'),
                html.Div("View market prices and charts", className='tooltip')
            ], className='nav-item'),
            html.Div([
                html.Button("Traders Portfolios", id='nav-traders-comparison', className='nav-button'),
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

        # Main content area with reorganized layout
        html.Div([
            # Market Overview Tab
            html.Div([
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
            ], id='market-overview-tab'),
            
            # Traders Portfolio Tab
            html.Div([
                # Signals Table with filter
                html.Div([
                    html.Div([
                        html.H3("Recent Signals"),
                        dcc.Dropdown(
                            id='signal-filter',
                            options=[
                                {'label': 'All Signals', 'value': 'all'},
                                {'label': 'Buy Signals', 'value': 'buy'},
                                {'label': 'Sell Signals', 'value': 'sell'},
                                {'label': 'Hold Signals', 'value': 'hold'}
                            ],
                            value='all',
                            className='signal-filter'
                        )
                    ], className='signals-header'),
                    html.Div(id='signals-table', className='signals-table-container')
                ], className='signals-section'),
                
                # Performance cards with sorting
                html.Div([
                    html.Div([
                        html.H3("Trader Performance"),
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
                
                # Agent Discussions Panel with filters
                html.Div([
                    html.Div([
                        html.H3("Agent Discussions"),
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
                ], className='discussions-panel')
            ], id='traders-portfolio-tab', style={'display': 'none'})
        ], className='main-content'),
        
        # Memory status indicator with animation
        html.Div([
            html.Span("💾", className="icon"),
            html.Span("Memory system active", id="memory-status-text"),
            html.Div(className="pulse-ring")
        ], id="memory-status", className="memory-status")
    ], className='dashboard-container')
    
    @app.callback(
        [Output('market-overview-tab', 'style'),
         Output('traders-portfolio-tab', 'style')],
        [Input('nav-market-overview', 'n_clicks'),
         Input('nav-traders-comparison', 'n_clicks')]
    )
    def toggle_tabs(market_clicks, traders_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return {'display': 'block'}, {'display': 'none'}
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'nav-market-overview':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
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
            recent_signals = trading_system.signals_history[-10:]  # Get last 10 signals
            
            # Apply filter
            if signal_filter != 'all':
                if signal_filter == 'buy':
                    recent_signals = [s for s in recent_signals if 'BUY' in s['signal']['action']]
                elif signal_filter == 'sell':
                    recent_signals = [s for s in recent_signals if 'SELL' in s['signal']['action']]
                elif signal_filter == 'hold':
                    recent_signals = [s for s in recent_signals if 'HOLD' in s['signal']['action']]
            
            signals_table = html.Div([
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
                            html.Td(
                                signal['signal']['action'],
                                className=f"signal-{signal['signal']['action'].lower()}"
                            ),
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
                        ], className='signal-row') for signal in reversed(recent_signals)
                    ])
                ], className='signals-table-content')
            ])
            
            return charts, market_overview, signals_table
            
        except Exception as e:
            print(f"Error updating trading view: {str(e)}")
            return [], html.Div("Error loading market data"), html.Div("Error loading signals")
    
    @app.callback(
        Output('traders-performance-cards', 'children'),
        [Input('auto-refresh-interval', 'n_intervals')]
    )
    def update_traders_comparison(n):
        """Update the traders comparison view."""
        try:
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
                    metrics = agent.wallet.get_performance_metrics(current_prices)
                    
                    # Get holdings that have non-zero amounts
                    holdings_display = []
                    for symbol, holding_data in metrics['holdings_with_prices'].items():
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
                    for trade in agent.wallet.trades_history[-5:]:  # Get last 5 trades
                        if isinstance(trade, dict) and 'symbol' in trade:
                            trades_history.append({
                                'symbol': trade['symbol'],
                                'is_buy': trade.get('action', '').upper() == 'BUY',
                                'amount': trade.get('amount_crypto', 0),
                                'price': trade.get('price', 0),
                                'value': trade.get('amount_usdt', 0)
                            })
                    
                    # Calculate goal progress
                    goal_progress = (metrics['total_value_usdt'] / 100.0) * 100
                    goal_status = "Goal Reached! 🏆" if goal_progress >= 100 else f"{goal_progress:.1f}% to $100"
                    
                    performance_data.append({
                        'name': agent.name,
                        'personality': agent.get_personality_traits()['personality'],
                        'total_value': metrics['total_value_usdt'],
                        'usdt_balance': metrics['balance_usdt'],
                        'crypto_value': metrics['total_value_usdt'] - metrics['balance_usdt'],
                        'holdings': holdings_display,
                        'trades': trades_history,
                        'goal_progress': goal_progress,
                        'goal_status': goal_status
                    })
                except Exception as e:
                    print(f"Error getting performance data for {agent.name}: {str(e)}")
                    continue
            
            # Sort by total value
            performance_data.sort(key=lambda x: x['total_value'], reverse=True)
            
            # Create performance cards
            cards = []
            for data in performance_data:
                card = html.Div([
                    html.Div([
                        html.H3([
                            html.Span(data['name'].replace(' AI', ''), className='agent-name'),
                            html.Span(data['personality'], className=f"agent-badge {data['personality'].lower()}")
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
                            html.H4("Recent Trades", className='trades-title'),
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
                            html.Div(data['goal_status'], className='goal-status')
                        ], className='goal-container')
                    ], className='card-content')
                ], className='performance-card')
                cards.append(card)
            
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
                                disc['timestamp'].strftime('%H:%M:%S'),
                                className='discussion-time'
                            )
                        ]),
                        html.Div(className='discussion-indicator')
                    ], className='discussion-header'),
                    html.Div([
                        html.Div([
                            html.Span(
                                message.split(':')[0],
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
            return dash.no_update
    
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
            # Make initial BTC purchase
            trading_system._make_initial_btc_purchase()
            return "All traders reset successfully"
        except Exception as e:
            print(f"Error resetting traders: {str(e)}")
            return "Error resetting traders"
    
    return app

if __name__ == '__main__':
    # Create trading system instance
    trading_system = TradingSystem()
    
    # Initialize the system
    trading_system._load_state()  # Load previous state if available
    
    # Start the trading system in a background thread
    trading_thread = Thread(target=trading_system.run)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Create and run the dashboard
    app = create_dashboard(trading_system)
    app.run_server(debug=True) 