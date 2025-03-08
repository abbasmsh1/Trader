import os
from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta
import time
from data.market_data import MarketDataFetcher
from agents.value_investor import ValueInvestor
from agents.tech_disruptor import TechDisruptor
from agents.trend_follower import TrendFollower
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from threading import Thread, Lock
import plotly.io as pio
from config.settings import DEFAULT_SYMBOLS, SYSTEM_PARAMS, TIMEFRAMES
import ta

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
            
            # Meme coins
            'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT',
            
            # Alt coins
            'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT', 'XRP/USDT',
            'ATOM/USDT', 'UNI/USDT', 'AAVE/USDT', 'NEAR/USDT', 'ARB/USDT'
        ]
        
        self.symbols = symbols or default_symbols
        
        # Initialize trading agents with different personalities
        self.agents = [
            ValueInvestor(name="Warren Buffett AI", timeframe='1d'),  # ValueInvestor doesn't accept risk_tolerance in __init__
            TechDisruptor(name="Elon Musk AI", timeframe='1h'),  # TechDisruptor has fixed risk_tolerance
            TrendFollower(name="Technical Trader", risk_tolerance=0.9, timeframe='4h')  # TrendFollower accepts risk_tolerance
        ]
        
        self.signals_history = []
        self.market_data_cache = {}
        self.last_update = {}
        self.auto_trading_enabled = True  # Enable auto-trading by default
        
        # Holdings history for each trader
        self.holdings_history = {agent.name: [] for agent in self.agents}
        
        # Thread safety
        self.lock = Lock()
        
        # Load saved state if available
        self._load_state()
        
        # Make initial BTC purchase for all agents if no state was loaded
        if not self._state_loaded:
            self._make_initial_btc_purchase()
            
    def _load_state(self):
        """Load saved state from disk if available."""
        self._state_loaded = False
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
                self._state_loaded = True
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
                        'wallet_metrics': agent.wallet.get_performance_metrics({'BTC': btc_price}),
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
        save_interval = 300  # Save state every 5 minutes
        
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
                
                # Save state periodically
                current_time = time.time()
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
                
                # Calculate crypto holdings value
                crypto_value = metrics['total_value_usdt'] - metrics['balance_usdt']
                
                # Record holdings snapshot
                holdings_snapshot = {
                    'timestamp': timestamp,
                    'total_value_usdt': metrics['total_value_usdt'],
                    'balance_usdt': metrics['balance_usdt'],
                    'crypto_value_usdt': crypto_value,
                    'holdings': {
                        symbol: {
                            'amount': amount,
                            'price_usdt': current_prices.get(symbol, 0),
                            'value_usdt': amount * current_prices.get(symbol, 0)
                        }
                        for symbol, amount in metrics['holdings'].items()
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

def create_dashboard(trading_system: TradingSystem):
    """
    Create a modern and interactive Dash dashboard with multiple coin pairs.
    """
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True
    )
    
    # Define theme colors
    colors = {
        'background': '#1a1a1a',  # Dark background
        'text': '#ffffff',        # White text
        'primary': '#2196F3',     # Blue primary color
        'secondary': '#f50057',   # Pink secondary color
        'success': '#4CAF50',     # Green success color
        'warning': '#ff9800',     # Orange warning color
        'error': '#f44336'        # Red error color
    }
    
    # Define colors for different actions
    action_colors = {
        'STRONG_BUY': '#00ff00',    # Bright green
        'BUY': '#4caf50',           # Regular green
        'SCALE_IN': '#8bc34a',      # Light green
        'WATCH': '#ffc107',         # Amber
        'HOLD': '#9e9e9e',          # Grey
        'SCALE_OUT': '#ff9800',     # Orange
        'SELL': '#f44336',          # Red
        'STRONG_SELL': '#d32f2f'    # Dark red
    }
    
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("AI Crypto Trading Dashboard", className='dashboard-title'),
            html.P("Trading bots competing to turn $20 into $100", className='dashboard-subtitle')
        ], className='header'),
        
        # Navigation
        html.Div([
            html.Button("Trading View", id='nav-trading-view', className='nav-button active'),
            html.Button("Traders Portfolios", id='nav-traders-comparison', className='nav-button')
        ], className='nav-container'),
        
        # Main content
        html.Div(id='page-content'),
        
        # Memory status indicator
        html.Div([
            html.Span("💾", className="icon"),
            html.Span("Memory system active", id="memory-status-text")
        ], id="memory-status", className="memory-status"),
        
        # Interval component for auto-refresh
        dcc.Interval(
            id='interval-component',
            interval=30 * 1000,  # 30 seconds
            n_intervals=0
        ),
        
        # Interval for memory status updates
        dcc.Interval(
            id='memory-save-interval',
            interval=300 * 1000,  # 5 minutes
            n_intervals=0
        )
    ], className='container')
    
    @app.callback(
        [Output('page-content', 'children'),
         Output('nav-trading-view', 'className'),
         Output('nav-traders-comparison', 'className')],
        [Input('nav-trading-view', 'n_clicks'),
         Input('nav-traders-comparison', 'n_clicks')]
    )
    def render_content(trading_view_clicks, traders_comparison_clicks):
        """Render the appropriate content based on navigation clicks."""
        ctx = callback_context
        
        if not ctx.triggered:
            button_id = 'nav-trading-view'  # Default view
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'nav-trading-view':
            return create_trading_view(trading_system), 'nav-button active', 'nav-button'
        elif button_id == 'nav-traders-comparison':
            return create_traders_comparison(trading_system), 'nav-button', 'nav-button active'
            
        # Default fallback
        return create_trading_view(trading_system), 'nav-button active', 'nav-button'
    
    def create_trading_view(trading_system):
        """Create the main trading view layout."""
        # Group symbols by type (USDT pairs and coin-to-coin pairs)
        usdt_pairs = [s for s in trading_system.symbols if s.endswith('/USDT')]
        coin_pairs = [s for s in trading_system.symbols if not s.endswith('/USDT')]
        
        return html.Div([
            # Main content
            html.Div([
                # Left panel - Controls
                html.Div([
                    html.Div([
                        html.H3("Trading Controls", className='panel-title'),
                        
                        # Display the symbols as tags instead of dropdowns
                        html.Div([
                            html.Label("USDT Pairs", className='section-label'),
                            html.Div([
                                html.Span(s.replace('/USDT', ''), className='coin-tag usdt-pair')
                                for s in usdt_pairs
                            ], className='coin-tags-container')
                        ], className='coin-section'),
                        
                        html.Div([
                            html.Label("Coin-to-Coin Pairs", className='section-label'),
                            html.Div([
                                html.Span(s, className='coin-tag coin-pair')
                                for s in coin_pairs
                            ], className='coin-tags-container')
                        ], className='coin-section'),
                        
                        html.Label("Timeframe"),
                        dcc.Dropdown(
                            id='timeframe-dropdown',
                            options=[{'label': v, 'value': k} for k, v in TIMEFRAMES.items()],
                            value='1h',
                            className='dropdown'
                        ),
                        html.Label("Technical Indicators"),
                        dcc.Checklist(
                            id='indicator-checklist',
                            options=[
                                {'label': 'Moving Averages', 'value': 'SMA'},
                                {'label': 'RSI', 'value': 'RSI'},
                                {'label': 'MACD', 'value': 'MACD'},
                                {'label': 'Bollinger Bands', 'value': 'BB'}
                            ],
                            value=['SMA', 'RSI'],
                            className='indicator-checklist'
                        ),
                        html.Label("Chart Style"),
                        dcc.RadioItems(
                            id='chart-style',
                            options=[
                                {'label': 'Candlestick', 'value': 'candlestick'},
                                {'label': 'Line', 'value': 'line'},
                                {'label': 'OHLC', 'value': 'ohlc'}
                            ],
                            value='candlestick',
                            className='chart-style-radio'
                        ),
                        html.Div([
                            html.Label("Auto Refresh"),
                            dcc.Checklist(
                                id='auto-refresh-switch',
                                options=[{'label': 'Enabled', 'value': 'enabled'}],
                                value=['enabled'],
                                className='auto-refresh-toggle'
                            )
                        ], className='switch-container'),
                        html.Div([
                            html.Label("Auto Trading"),
                            dcc.Checklist(
                                id='auto-trading-switch',
                                options=[{'label': 'Enabled', 'value': 'enabled'}],
                                value=['enabled'] if trading_system.auto_trading_enabled else [],
                                className='auto-trading-toggle'
                            )
                        ], className='switch-container'),
                        html.Div([
                            html.Label("Chart Layout"),
                            dcc.RadioItems(
                                id='layout-radio',
                                options=[
                                    {'label': '2x2 Grid', 'value': '2x2'},
                                    {'label': '1x4 Row', 'value': '1x4'},
                                    {'label': '2x3 Grid', 'value': '2x3'},
                                    {'label': '3x3 Grid', 'value': '3x3'},
                                    {'label': '4x4 Grid', 'value': '4x4'}
                                ],
                                value='3x3',
                                className='layout-radio'
                            )
                        ], className='layout-container')
                    ], className='control-panel')
                ], className='left-panel'),
                
                # Right panel - Charts and Data
                html.Div([
                    # Charts Container
                    html.Div(id='multi-chart-container', className='chart-grid'),
                    
                    # Market Overview
                    html.Div([
                        html.H3("Market Overview", className='panel-title'),
                        html.Div(id='market-overview', className='market-overview')
                    ], className='chart-panel'),
                    
                    # Trading Signals
                    html.Div([
                        html.H3("Latest Trading Signals", className='panel-title'),
                        html.Div(id='signals-table', className='signals-table')
                    ], className='signals-panel')
                ], className='right-panel')
            ], className='main-content')
        ], className='trading-view')
    
    def create_traders_comparison(trading_system):
        """Create the traders comparison view."""
        return html.Div([
            # Portfolio Overview Section
            html.Div([
                html.H2("Traders Portfolio Overview", className='section-title'),
                
                # Total Holdings Summary Cards
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Total Holdings (USDT)", className='holdings-title'),
                            html.Div([
                                create_holdings_summary(trading_system)
                            ], className='holdings-summary-container')
                        ], className='card-content')
                    ], className='holdings-summary-card')
                ], className='holdings-summary-section'),
                
                # Performance metrics
                html.Div(id='traders-performance-cards', className='performance-cards-container'),
                
                # Portfolio value chart
                html.Div([
                    html.H3("Portfolio Value Comparison", className='subsection-title'),
                    html.Div(id='portfolio-value-chart', className='portfolio-chart')
                ], className='chart-section'),
                
                # Holdings comparison
                html.Div([
                    html.H3("Holdings Breakdown", className='subsection-title'),
                    html.Div(id='holdings-comparison', className='holdings-comparison')
                ], className='chart-section'),
                
                # Trade activity
                html.Div([
                    html.H3("Trade Activity", className='subsection-title'),
                    html.Div(id='trade-activity-comparison', className='trade-activity')
                ], className='chart-section'),
                
                # Trade history section
                create_trade_history(trading_system)
            ], className='traders-comparison-view')
        ])
        
    def create_holdings_summary(trading_system):
        """Create a summary of total holdings for each trader."""
        # Get current prices for all symbols
        current_prices = {}
        for symbol in trading_system.symbols:
            try:
                if symbol.endswith('/USDT'):
                    df = trading_system.get_market_data(symbol)
                    if not df.empty:
                        base_currency = symbol.split('/')[0]
                        current_prices[base_currency] = float(df['close'].iloc[-1])
            except Exception as e:
                print(f"Error getting price for {symbol}: {str(e)}")
        
        # Get performance data for each agent
        performance_data = []
        for agent in trading_system.agents:
            # Get wallet metrics
            metrics = agent.wallet.get_performance_metrics(current_prices)
            total_value = metrics['total_value_usdt']
            usdt_balance = metrics['balance_usdt']
            
            # Calculate crypto holdings value
            crypto_value = total_value - usdt_balance
            
            # Calculate goal progress
            goal_progress = total_value / 100.0
            goal_status = "Goal Reached! 🏆" if goal_progress >= 1.0 else f"{goal_progress*100:.1f}% to $100"
            
            # Calculate crypto allocation
            crypto_percentage = (crypto_value / total_value) * 100 if total_value > 0 else 0
            usdt_percentage = (usdt_balance / total_value) * 100 if total_value > 0 else 0
            
            # Store the data
            performance_data.append({
                'agent': agent,
                'name': agent.name,
                'personality': agent.get_personality_traits()['personality'],
                'total_value': total_value,
                'usdt_balance': usdt_balance,
                'crypto_value': crypto_value,
                'crypto_percentage': crypto_percentage,
                'usdt_percentage': usdt_percentage,
                'goal_progress': goal_progress,
                'goal_status': goal_status
            })
        
        # Sort by total value (highest first)
        performance_data.sort(key=lambda x: x['total_value'], reverse=True)
        
        # Create summary cards
        summary_cards = []
        for i, data in enumerate(performance_data):
            # Create the summary card
            card = html.Div([
                html.Div([
                    html.Div([
                        html.Span(data['name'].replace(' AI', ''), className='agent-name'),
                        html.Span(data['personality'], 
                                 className=f"agent-badge {data['personality'].lower()}")
                    ], className='summary-header'),
                    
                    html.Div([
                        html.Div(f"${data['total_value']:.2f}", className='total-value'),
                        html.Div("Total Wallet Value", className='value-label')
                    ], className='value-container'),
                    
                    # Wallet composition visualization
                    html.Div([
                        html.Div("Wallet Composition", className='composition-title'),
                        html.Div(className='composition-bar', children=[
                            html.Div(
                                className='usdt-portion',
                                style={'width': f"{data['usdt_percentage']}%"}
                            ),
                            html.Div(
                                className='crypto-portion',
                                style={'width': f"{data['crypto_percentage']}%"}
                            )
                        ]),
                        html.Div(className='composition-legend', children=[
                            html.Div([
                                html.Span(className='usdt-dot'),
                                html.Span(f"USDT: ${data['usdt_balance']:.2f} ({data['usdt_percentage']:.1f}%)")
                            ], className='legend-item'),
                            html.Div([
                                html.Span(className='crypto-dot'),
                                html.Span(f"Crypto: ${data['crypto_value']:.2f} ({data['crypto_percentage']:.1f}%)")
                            ], className='legend-item')
                        ])
                    ], className='wallet-composition'),
                    
                    html.Div([
                        html.Div(className='goal-progress-bar', children=[
                            html.Div(
                                className='goal-progress-fill',
                                style={'width': f"{min(100, data['goal_progress']*100)}%"}
                            )
                        ]),
                        html.Div(data['goal_status'], className='goal-status')
                    ], className='goal-container')
                ], className='summary-content')
            ], className=f"holdings-summary-card {i == 0 and 'leading' or ''}")
            
            summary_cards.append(card)
        
        return html.Div(summary_cards, className='holdings-summary-grid')
    
    def create_trade_history(trading_system):
        """Create a component to display trade history for each trader."""
        trade_history_cards = []
        
        for agent in trading_system.agents:
            # Get trade history from wallet
            trades = agent.wallet.trades_history
            
            if not trades:
                # No trades yet
                card = html.Div([
                    html.Div([
                        html.H3([
                            html.Span(agent.name.replace(' AI', ''), className='agent-name'),
                            html.Span(agent.get_personality_traits()['personality'], 
                                     className=f"agent-badge {agent.get_personality_traits()['personality'].lower()}")
                        ]),
                        html.Div("No trades executed yet", className='no-trades-message')
                    ], className='card-content')
                ], className='trade-history-card')
            else:
                # Sort trades by timestamp (newest first)
                sorted_trades = sorted(trades, key=lambda x: x['timestamp'], reverse=True)
                
                # Create trade list
                trade_items = []
                for trade in sorted_trades[:10]:  # Show only the 10 most recent trades
                    # Format timestamp
                    timestamp = trade['timestamp']
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        time_str = "Unknown time"
                    
                    # Determine trade class based on action
                    trade_class = 'trade-buy' if trade['action'] == 'BUY' else 'trade-sell'
                    
                    # Format amounts
                    amount_usdt = f"${trade['amount_usdt']:.2f}"
                    amount_crypto = f"{trade['amount_crypto']:.8f}"
                    
                    # Create trade item
                    trade_item = html.Div([
                        html.Div([
                            html.Span(trade['action'], className=f'trade-action {trade_class}'),
                            html.Span(trade['symbol'], className='trade-symbol')
                        ], className='trade-header'),
                        html.Div([
                            html.Div([
                                html.Span("Amount: ", className='trade-label'),
                                html.Span(f"{amount_crypto} @ ${trade['price']:.2f}", className='trade-value')
                            ], className='trade-detail'),
                            html.Div([
                                html.Span("Value: ", className='trade-label'),
                                html.Span(amount_usdt, className='trade-value')
                            ], className='trade-detail'),
                            html.Div([
                                html.Span("Time: ", className='trade-label'),
                                html.Span(time_str, className='trade-value trade-time')
                            ], className='trade-detail')
                        ], className='trade-details')
                    ], className=f'trade-item {trade_class}')
                    
                    trade_items.append(trade_item)
                
                # Create card with trades
                card = html.Div([
                    html.Div([
                        html.H3([
                            html.Span(agent.name.replace(' AI', ''), className='agent-name'),
                            html.Span(agent.get_personality_traits()['personality'], 
                                     className=f"agent-badge {agent.get_personality_traits()['personality'].lower()}")
                        ]),
                        html.Div(f"Total trades: {len(trades)}", className='trade-count'),
                        html.Div(trade_items, className='trade-list')
                    ], className='card-content')
                ], className='trade-history-card')
            
            trade_history_cards.append(card)
        
        return html.Div([
            html.H2("Trade History", className='section-title'),
            html.Div(trade_history_cards, className='trade-history-container')
        ], className='trade-history-section')
    
    # Callback to update traders comparison view
    @app.callback(
        [Output('traders-performance-cards', 'children'),
         Output('portfolio-value-chart', 'children'),
         Output('holdings-comparison', 'children'),
         Output('trade-activity-comparison', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_traders_comparison(n):
        """Update the traders comparison view."""
        try:
            # Get current prices for all symbols
            current_prices = {}
            for symbol in trading_system.symbols:
                try:
                    if symbol.endswith('/USDT'):
                        df = trading_system.get_market_data(symbol)
                        if not df.empty:
                            base_currency = symbol.split('/')[0]
                            current_prices[base_currency] = float(df['close'].iloc[-1])
                except Exception as e:
                    print(f"Error getting price for {symbol}: {str(e)}")
            
            # Get performance metrics for each agent
            performance_data = []
            for agent in trading_system.agents:
                metrics = agent.wallet.get_performance_metrics(current_prices)
                
                # Calculate progress toward $100 goal
                goal_progress = metrics['total_value_usdt'] / 100.0
                goal_status = "Reached!" if goal_progress >= 1.0 else f"{goal_progress*100:.1f}%"
                
                # Calculate holdings with USDT values
                holdings_with_value = []
                for symbol, amount in metrics['holdings'].items():
                    if symbol in current_prices:
                        usdt_value = amount * current_prices[symbol]
                        percentage = (usdt_value / metrics['total_value_usdt']) * 100 if metrics['total_value_usdt'] > 0 else 0
                        holdings_with_value.append({
                            'symbol': symbol,
                            'amount': amount,
                            'price': current_prices[symbol],
                            'usdt_value': usdt_value,
                            'percentage': percentage
                        })
                
                # Sort holdings by USDT value (highest first)
                holdings_with_value.sort(key=lambda x: x['usdt_value'], reverse=True)
                
                performance_data.append({
                    'agent': agent.name,
                    'personality': agent.get_personality_traits()['personality'],
                    'total_value': metrics['total_value_usdt'],
                    'balance_usdt': metrics['balance_usdt'],
                    'holdings': metrics['holdings'],
                    'holdings_with_value': holdings_with_value,
                    'total_return': metrics['total_return_pct'],
                    'trade_count': metrics['trade_count'],
                    'goal_progress': goal_progress,
                    'goal_status': goal_status
                })
            
            # Sort by total value (best performing first)
            performance_data.sort(key=lambda x: x['total_value'], reverse=True)
            
            # Create performance cards
            performance_cards = html.Div([
                html.Div([
                    html.Div([
                        html.H3([
                            html.Span(data['agent'].replace(' AI', ''), className='agent-name'),
                            html.Span(data['personality'], className=f"agent-badge {data['personality'].lower()}")
                        ]),
                        html.Div([
                            html.Div(f"${data['total_value']:.2f}", className='metric-value'),
                            html.Div("Total Value", className='metric-label')
                        ], className='metric'),
                        html.Div([
                            html.Div(f"{data['total_return']:.2f}%", 
                                    className=f"metric-value {'positive' if data['total_return'] > 0 else 'negative'}"),
                            html.Div("Return", className='metric-label')
                        ], className='metric'),
                        html.Div([
                            html.Div(f"{data['trade_count']}", className='metric-value'),
                            html.Div("Trades", className='metric-label')
                        ], className='metric'),
                        html.Div([
                            html.Div(className='progress-bar-container', children=[
                                html.Div(
                                    className='progress-bar',
                                    style={'width': f"{min(100, data['goal_progress']*100)}%"}
                                ),
                                html.Span(
                                    data['goal_status'], 
                                    className=f"progress-text {'' if data['goal_progress'] < 1.0 else 'goal-reached'}"
                                )
                            ]),
                            html.Div("Goal Progress ($100)", className='metric-label')
                        ], className='metric goal-metric'),
                    ], className='card-content')
                ], className=f"performance-card {'winner' if i == 0 else ''}") 
                for i, data in enumerate(performance_data)
            ], className='performance-cards-container')
            
            # Create portfolio value chart
            fig = go.Figure()
            
            for data in performance_data:
                # Add a horizontal bar for each agent
                fig.add_trace(go.Bar(
                    y=[data['agent'].replace(' AI', '')],
                    x=[data['total_value']],
                    orientation='h',
                    name=data['agent'].replace(' AI', ''),
                    marker=dict(
                        color='#4CAF50' if data['goal_progress'] >= 1.0 else '#2196F3',
                        line=dict(color='rgba(0,0,0,0)', width=0)
                    ),
                    hovertemplate='$%{x:.2f}<extra></extra>'
                ))
                
                # Add a marker for the $100 goal
                fig.add_shape(
                    type="line",
                    x0=100, x1=100,
                    y0=-0.5, y1=len(performance_data) - 0.5,
                    line=dict(color="#FF9800", width=2, dash="dash"),
                )
                
            fig.update_layout(
                title="Portfolio Value Comparison",
                template='plotly_dark',
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Value (USDT)",
                showlegend=False,
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.1)'
                ),
                annotations=[
                    dict(
                        x=100,
                        y=len(performance_data),
                        xref="x",
                        yref="y",
                        text="$100 Goal",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-30,
                        font=dict(color="#FF9800")
                    )
                ]
            )
            
            portfolio_chart = html.Div([
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False}
                )
            ], className='portfolio-chart')
            
            # Create enhanced holdings comparison with USDT values
            holdings_comparison = html.Div([
                html.Div([
                    html.Div([
                        # Header with agent name and total value
                        html.Div([
                            html.H4(data['agent'].replace(' AI', ''), className='holdings-card-title'),
                            html.Div(f"Total Value: ${data['total_value']:.2f}", className='holdings-total-value')
                        ], className='holdings-card-header'),
                        
                        # Wallet composition visualization
                        html.Div([
                            html.Div(className='wallet-bar', children=[
                                html.Div(
                                    className='usdt-bar',
                                    style={'width': f"{data['balance_usdt'] / data['total_value'] * 100 if data['total_value'] > 0 else 0}%"}
                                ),
                                html.Div(
                                    className='crypto-bar',
                                    style={'width': f"{(data['total_value'] - data['balance_usdt']) / data['total_value'] * 100 if data['total_value'] > 0 else 0}%"}
                                )
                            ]),
                            html.Div(className='wallet-legend', children=[
                                html.Div([
                                    html.Span("USDT:", className='legend-label'),
                                    html.Span(f"${data['balance_usdt']:.2f}", className='legend-value')
                                ], className='legend-row'),
                                html.Div([
                                    html.Span("Crypto:", className='legend-label'),
                                    html.Span(f"${(data['total_value'] - data['balance_usdt']):.2f}", className='legend-value')
                                ], className='legend-row')
                            ])
                        ], className='wallet-summary'),
                        
                        # Detailed Holdings with USDT values
                        html.Div([
                            html.H5("Crypto Holdings", className='holdings-subtitle'),
                            html.Table([
                                html.Thead(
                                    html.Tr([
                                        html.Th("Asset"),
                                        html.Th("Amount"),
                                        html.Th("Price (USDT)"),
                                        html.Th("Value (USDT)"),
                                        html.Th("% of Portfolio")
                                    ])
                                ),
                                html.Tbody([
                                    html.Tr([
                                        html.Td([
                                            get_coin_category_span(holding['symbol']),
                                            html.Span(holding['symbol'])
                                        ], className='asset-cell'),
                                        html.Td(f"{holding['amount']:.8f}", className='amount-cell'),
                                        html.Td(f"${holding['price']:.2f}", className='price-cell'),
                                        html.Td(f"${holding['usdt_value']:.2f}", className='value-cell'),
                                        html.Td([
                                            html.Div(className='percentage-bar-container', children=[
                                                html.Div(
                                                    className='percentage-bar',
                                                    style={'width': f"{min(100, holding['percentage'])}%"}
                                                ),
                                                html.Span(f"{holding['percentage']:.1f}%", className='percentage-text')
                                            ])
                                        ], className='percentage-cell')
                                    ], className=f"holding-row {i % 2 == 0 and 'even' or 'odd'}") 
                                    for i, holding in enumerate(data['holdings_with_value'])
                                ] if data['holdings_with_value'] else [
                                    html.Tr([
                                        html.Td(
                                            "No crypto holdings", 
                                            colSpan=5, 
                                            className='no-holdings-message'
                                        )
                                    ])
                                ])
                            ], className='holdings-table')
                        ], className='holdings-detail')
                    ], className='holdings-card')
                    for data in performance_data
                ], className='holdings-grid')
            ], className='holdings-comparison')
            
            # Create trade activity comparison
            trade_activity = html.Div([
                html.Div(id='trade-activity-content', children=[
                    html.Div("Loading trade activity...", className='loading-message')
                ])
            ], className='trade-activity-comparison')
            
            return performance_cards, portfolio_chart, holdings_comparison, trade_activity
            
        except Exception as e:
            print(f"Error updating traders comparison: {str(e)}")
            return (
                html.Div(f"Error loading performance data: {str(e)}"),
                html.Div("Error loading portfolio chart"),
                html.Div("Error loading holdings comparison"),
                html.Div("Error loading trade activity")
            )
            
    def get_coin_category_span(symbol):
        """Get a span element with the appropriate coin category styling."""
        if symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']:
            category = "major"
            emoji = "💎"
        elif symbol in ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK']:
            category = "meme"
            emoji = "🚀"
        else:
            category = "alt"
            emoji = "⭐"
            
        return html.Span(emoji, className=f"coin-category-icon {category}-coin")
    
    def create_signals_table(signals):
        """Create a table of trading signals."""
        if not signals:
            return html.Div("No signals available", className='no-data')
            
        return html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Agent"),
                    html.Th("Quote"),
                    html.Th("Symbol"),
                    html.Th("Action"),
                    html.Th("Confidence"),
                    html.Th("Reason"),
                    html.Th("Wallet Value"),
                    html.Th("Goal Progress"),
                    html.Th("Trade Status")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td([
                            html.Span(signal['agent'].replace(' AI', ''), className='agent-name'),
                            html.Span(signal['personality'], className=f"agent-badge {signal['personality'].lower()}")
                        ]),
                        html.Td(
                            signal.get('strategy', {}).get('famous_quote', 'No quote available'),
                            className='trader-quote'
                        ),
                        html.Td(signal['symbol'].replace('/USDT', '')),
                        html.Td(
                            signal['signal']['action'],
                            className=f"action-{signal['signal']['action'].lower()}"
                        ),
                        html.Td(f"{signal['signal']['confidence']:.2f}"),
                        html.Td(signal['signal'].get('reason', 'No reason provided')),
                        html.Td(f"${signal['wallet_metrics']['total_value_usdt']:.2f}"),
                        html.Td([
                            html.Div(className='progress-bar-container', children=[
                                html.Div(
                                    className='progress-bar',
                                    style={'width': f"{min(100, signal['wallet_metrics']['total_value_usdt'])}%"}
                                ),
                                html.Span(f"{signal['wallet_metrics']['total_value_usdt']/100*100:.1f}%", className='progress-text')
                            ])
                        ]),
                        html.Td(
                            "✅ Executed" if signal.get('trade_executed', False) else 
                            "⏳ Pending" if signal['signal']['action'] not in ['HOLD', 'WATCH'] else 
                            "⏹️ No Action",
                            className=f"trade-status-{'executed' if signal.get('trade_executed', False) else 'pending' if signal['signal']['action'] not in ['HOLD', 'WATCH'] else 'no-action'}"
                        )
                    ]) for signal in signals
                ])
            ], className='signals-table-content')
        ], className='signals-table-container')
    
    def create_performance_metrics(performance_data):
        return html.Div([
            html.Div([
                html.H4(data['symbol']),
                html.Div([
                    html.Div(f"${data['price']:,.2f}", className='metric-value'),
                    html.Div(f"{data['change_24h']:+.2f}%", 
                            className=f"metric-change {'positive' if data['change_24h'] > 0 else 'negative'}")
                ], className='price-container'),
                html.Div([
                    html.Div(f"24h Volume: ${data['volume_24h']:,.0f}"),
                    html.Div(f"24h High: ${data['high_24h']:,.2f}"),
                    html.Div(f"24h Low: ${data['low_24h']:,.2f}")
                ], className='metrics-details')
            ], className='performance-card')
            for data in performance_data
        ], className='performance-grid')
    
    def create_market_overview(market_data):
        return html.Div([
            html.Div([
                html.H4(data['symbol']),
                html.Div([
                    html.Div(f"Volatility: {data['volatility']:.2f}%"),
                    html.Div(f"Avg Volume: ${data['avg_volume']:,.0f}"),
                    html.Div(f"Trend: {data['trend']}", 
                            className=f"trend-indicator {'bullish' if data['trend'] == 'Bullish' else 'bearish'}")
                ], className='market-details')
            ], className='market-card')
            for data in market_data
        ], className='market-grid')
    
    @app.callback(
        [Output('multi-chart-container', 'children'),
         Output('market-overview', 'children'),
         Output('signals-table', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('timeframe-dropdown', 'value'),
         Input('indicator-checklist', 'value'),
         Input('chart-style', 'value'),
         Input('layout-radio', 'value')],
        [State('auto-refresh-switch', 'value')]
    )
    def update_trading_view(n, timeframe, indicators, chart_style, layout, auto_refresh):
        """Update the trading view components."""
        print("\nUpdating trading view...")
        
        # Get all symbols directly from the trading system
        all_symbols = trading_system.symbols
        
        # Limit the number of symbols based on the layout to avoid performance issues
        max_charts = {
            '2x2': 4,
            '1x4': 4,
            '2x3': 6,
            '3x3': 9,
            '4x4': 16
        }.get(layout, 9)
        
        # Prioritize major coins, then meme coins, then alt coins
        major_coins = [s for s in all_symbols if s.split('/')[0] in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']]
        meme_coins = [s for s in all_symbols if s.split('/')[0] in ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK']]
        coin_pairs = [s for s in all_symbols if '/' in s and not s.endswith('/USDT')]
        other_coins = [s for s in all_symbols if s not in major_coins + meme_coins + coin_pairs]
        
        # Order the symbols by priority
        ordered_symbols = major_coins + meme_coins + coin_pairs + other_coins
        
        # Limit to max_charts
        display_symbols = ordered_symbols[:max_charts]
        
        print(f"Displaying {len(display_symbols)} out of {len(all_symbols)} symbols")
        print(f"Timeframe: {timeframe}")
        print(f"Indicators: {indicators}")
        print(f"Chart style: {chart_style}")
        print(f"Layout: {layout}")
        
        if not callback_context.triggered:
            print("No context triggered")
            raise PreventUpdate
        
        if 'enabled' not in auto_refresh and callback_context.triggered[0]['prop_id'] == 'interval-component.n_intervals':
            print("Auto-refresh disabled")
            raise PreventUpdate
        
        if not display_symbols:
            print("No symbols selected")
            return [], html.Div("No symbols selected"), html.Div("No signals available")
        
        try:
            # Create charts
            charts = []
            market_overview_data = []
            
            for symbol in display_symbols:
                try:
                    print(f"Processing {symbol}...")
                    df = trading_system.get_market_data(symbol)
                    if df.empty:
                        print(f"No data available for {symbol}")
                        continue
                    
                    fig = go.Figure()
                    
                    # Add price data
                    if chart_style == 'candlestick':
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name=symbol
                        ))
                    elif chart_style == 'line':
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['close'],
                            mode='lines',
                            name=symbol
                        ))
                    elif chart_style == 'ohlc':
                        fig.add_trace(go.Ohlc(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name=symbol
                        ))
                    
                    # Add indicators
                    if 'SMA' in indicators:
                        sma20 = ta.trend.sma_indicator(df['close'], window=20)
                        sma50 = ta.trend.sma_indicator(df['close'], window=50)
                        fig.add_trace(go.Scatter(x=df.index, y=sma20, name='SMA 20', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=df.index, y=sma50, name='SMA 50', line=dict(color='blue')))
                    
                    if 'RSI' in indicators:
                        rsi = ta.momentum.rsi(df['close'])
                        fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', yaxis='y2'))
                        fig.update_layout(
                            yaxis2=dict(
                                title="RSI",
                                overlaying="y",
                                side="right",
                                range=[0, 100]
                            )
                        )
                        
                    if 'MACD' in indicators:
                        macd = ta.trend.macd(df['close'])
                        macd_signal = ta.trend.macd_signal(df['close'])
                        macd_diff = ta.trend.macd_diff(df['close'])
                        fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=df.index, y=macd_signal, name='Signal', line=dict(color='orange')))
                        fig.add_trace(go.Bar(x=df.index, y=macd_diff, name='Histogram', marker_color='green'))
                    
                    if 'BB' in indicators:
                        bb = ta.volatility.BollingerBands(df['close'])
                        fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband(),
                                               name='BB Upper', line=dict(dash='dash')))
                        fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband(),
                                               name='BB Lower', line=dict(dash='dash')))
                    
                    # Update layout
                    # For coin-to-coin pairs, show the ratio in the title
                    if not symbol.endswith('/USDT'):
                        title = f'{symbol} Ratio Chart'
                    else:
                        title = f'{symbol} Price Chart'
                        
                    fig.update_layout(
                        title=title,
                        template='plotly_dark',
                        height=400,
                        xaxis_rangeslider_visible=False,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    # Create chart container based on layout
                    chart_container_class = f'chart-container-{layout}'
                    chart = html.Div([
                        dcc.Graph(
                            figure=fig,
                            config={'displayModeBar': True}
                        )
                    ], className=chart_container_class)
                    
                    charts.append(chart)
                    
                    # Add market overview data
                    current_price = df['close'].iloc[-1]
                    price_change = ((current_price / df['close'].iloc[-2] - 1) * 100)
                    
                    # For coin-to-coin pairs, show the ratio
                    if not symbol.endswith('/USDT'):
                        base, quote = symbol.split('/')
                        display_symbol = f"{base}/{quote}"
                        price_format = f"{current_price:.6f}"
                        coin_category = "ratio"
                    else:
                        # Determine coin category
                        base = symbol.split('/')[0]
                        if base in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']:
                            coin_category = "major"
                        elif base in ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK']:
                            coin_category = "meme"
                        else:
                            coin_category = "alt"
                            
                        display_symbol = base
                        price_format = f"${current_price:.2f}"
                    
                    market_overview_data.append({
                        'symbol': display_symbol,
                        'price': price_format,
                        'change': price_change,
                        'volume': df['volume'].iloc[-1],
                        'is_ratio': not symbol.endswith('/USDT'),
                        'category': coin_category
                    })
                    
                except Exception as e:
                    print(f"Error processing chart for {symbol}: {str(e)}")
                    continue
            
            if not charts:
                print("No charts created")
                return [], html.Div("No data available"), html.Div("No signals available")
            
            # Apply layout class to chart grid
            chart_grid_class = f'chart-grid-{layout}'
            chart_grid = html.Div(charts, className=chart_grid_class)
            
            # Create market overview table
            market_overview = html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Symbol"),
                        html.Th("Category"),
                        html.Th("Price"),
                        html.Th("24h Change"),
                        html.Th("Volume")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(data['symbol']),
                            html.Td(html.Span(data['category'].capitalize(), className=f"coin-category {data['category']}")),
                            html.Td(data['price']),
                            html.Td(
                                f"{data['change']:.2f}%",
                                className=f"{'positive' if data['change'] > 0 else 'negative'}"
                            ),
                            html.Td(f"{data['volume']:,.0f}" if not data['is_ratio'] else "N/A")
                        ]) for data in market_overview_data
                    ])
                ], className='market-overview-table')
            ])
            
            # Create signals table
            recent_signals = sorted(
                [s for s in trading_system.signals_history if s['symbol'] in all_symbols],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:10]
            
            signals_table = create_signals_table(recent_signals)
            
            print(f"Successfully created {len(charts)} charts")
            return chart_grid, market_overview, signals_table
            
        except Exception as e:
            print(f"Error updating trading view: {str(e)}")
            return [], html.Div(f"Error: {str(e)}"), html.Div("No signals available")
    
    # Callback to toggle auto-trading
    @app.callback(
        Output('auto-trading-switch', 'value'),
        [Input('auto-trading-switch', 'value')]
    )
    def toggle_auto_trading(value):
        """Toggle auto-trading based on the switch value."""
        enabled = 'enabled' in value if value else False
        trading_system.toggle_auto_trading(enabled)
        return value
    
    # Add custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>AI Crypto Trading Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                /* Custom CSS */
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    background-color: #1f2630;
                    color: #ffffff;
                }
                .container {
                    padding: 20px;
                }
                .header {
                    text-align: center;
                    padding: 20px;
                    background-color: #2b3c4e;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .dashboard-title {
                    margin: 0;
                    color: #ffffff;
                    font-size: 2.5em;
                }
                .dashboard-subtitle {
                    color: #a8b2c1;
                    margin: 10px 0 0 0;
                }
                .main-content {
                    display: flex;
                    gap: 20px;
                }
                .left-panel {
                    flex: 1;
                    max-width: 300px;
                }
                .right-panel {
                    flex: 3;
                }
                .control-panel, .chart-panel, .signals-panel {
                    background-color: #2b3c4e;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .panel-title {
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: #ffffff;
                }
                .dropdown {
                    margin-bottom: 20px;
                }
                .signals-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .signals-table th, .signals-table td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #3d4c5e;
                }
                .signal-buy {
                    color: #00ff9f;
                    font-weight: bold;
                }
                .signal-sell {
                    color: #ff4757;
                    font-weight: bold;
                }
                .signal-hold {
                    color: #ffa502;
                    font-weight: bold;
                }
                .confidence-container {
                    position: relative;
                    background-color: #283442;
                    border-radius: 10px;
                    height: 20px;
                    overflow: hidden;
                }
                .confidence-bar {
                    position: absolute;
                    height: 100%;
                    background-color: #00ff9f;
                    opacity: 0.3;
                }
                .confidence-container span {
                    position: relative;
                    padding: 0 10px;
                    line-height: 20px;
                }
                .switch-container {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-top: 20px;
                }
                .auto-refresh-toggle {
                    display: inline-flex;
                    align-items: center;
                }
                /* Style the checkbox to look like a toggle switch */
                .auto-refresh-toggle input[type="checkbox"] {
                    appearance: none;
                    width: 50px;
                    height: 24px;
                    background-color: #283442;
                    border-radius: 12px;
                    position: relative;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }
                .auto-refresh-toggle input[type="checkbox"]:checked {
                    background-color: #00ff9f;
                }
                .auto-refresh-toggle input[type="checkbox"]::before {
                    content: "";
                    position: absolute;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background-color: white;
                    top: 2px;
                    left: 2px;
                    transition: transform 0.3s;
                }
                .auto-refresh-toggle input[type="checkbox"]:checked::before {
                    transform: translateX(26px);
                }
                /* New styles for multi-chart layout */
                .chart-grid {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                
                .chart-grid-2x2 {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }
                
                .chart-grid-1x4 {
                    display: grid;
                    grid-template-columns: repeat(1, 1fr);
                    gap: 20px;
                }
                
                .chart-container-2x2 {
                    background-color: #2b3c4e;
                    border-radius: 10px;
                    padding: 15px;
                }
                
                .chart-container-1x4 {
                    background-color: #2b3c4e;
                    border-radius: 10px;
                    padding: 15px;
                }
                
                .layout-container {
                    margin-top: 20px;
                }
                
                .layout-radio {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    margin-top: 10px;
                }
                
                .no-data-message {
                    text-align: center;
                    padding: 20px;
                    background-color: #2b3c4e;
                    border-radius: 10px;
                    color: #a8b2c1;
                }
                /* Responsive design updates */
                @media (max-width: 1024px) {
                    .chart-grid-2x2 {
                        grid-template-columns: 1fr;
                    }
                
                    .main-content {
                        flex-direction: column;
                    }
                    
                    .left-panel {
                        max-width: none;
                    }
                }
                
                /* New styles for technical indicators */
                .indicator-checklist {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                    margin-bottom: 20px;
                }
                
                .chart-style-radio {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                    margin-bottom: 20px;
                }
                
                /* Performance metrics styles */
                .performance-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 10px;
                }
                
                .performance-card {
                    background-color: #283442;
                    border-radius: 8px;
                    padding: 15px;
                }
                
                .price-container {
                    display: flex;
                    align-items: baseline;
                    gap: 10px;
                    margin: 10px 0;
                }
                
                .metric-value {
                    font-size: 1.5em;
                    font-weight: bold;
                }
                
                .metric-change {
                    font-size: 1.1em;
                }
                
                .metric-change.positive {
                    color: #00ff9f;
                }
                
                .metric-change.negative {
                    color: #ff4757;
                }
                
                .metrics-details {
                    font-size: 0.9em;
                    color: #a8b2c1;
                }
                
                /* Market overview styles */
                .market-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 10px;
                }
                
                .market-card {
                    background-color: #283442;
                    border-radius: 8px;
                    padding: 15px;
                }
                
                .market-details {
                    margin-top: 10px;
                    font-size: 0.9em;
                }
                
                .trend-indicator {
                    font-weight: bold;
                    margin-top: 5px;
                }
                
                .trend-indicator.bullish {
                    color: #00ff9f;
                }
                
                .trend-indicator.bearish {
                    color: #ff4757;
                }
                
                /* Slider styles */
                .slider-container {
                    margin-top: 20px;
                }
                
                .refresh-slider {
                    margin-top: 10px;
                }
                
                /* Chart grid updates */
                .chart-grid-2x3 {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }
                
                .chart-container-2x3 {
                    background-color: #2b3c4e;
                    border-radius: 10px;
                    padding: 15px;
                }
                
                /* Responsive updates */
                @media (max-width: 1024px) {
                    .chart-grid-2x3 {
                        grid-template-columns: 1fr;
                    }
                
                    .performance-grid,
                    .market-grid {
                        grid-template-columns: 1fr;
                    }
                }
                
                /* Animation effects */
                .performance-card,
                .market-card,
                .chart-container-2x2,
                .chart-container-1x4,
                .chart-container-2x3 {
                    transition: transform 0.2s ease-in-out;
                }
                
                .performance-card:hover,
                .market-card:hover {
                    transform: translateY(-2px);
                }
                
                /* Loading states */
                .loading-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                }
                
                .loading-spinner {
                    border: 4px solid #283442;
                    border-top: 4px solid #00ff9f;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                /* Personality-based styles */
                .signal-commentary {
                    font-style: italic;
                    color: #a8b2c1;
                    font-size: 0.9em;
                }
                
                .agent-warren {
                    color: #4CAF50;  /* Value investing green */
                }
                
                .agent-elon {
                    color: #2196F3;  /* Tech blue */
                }
                
                .agent-technical {
                    color: #FFC107;  /* Technical analysis gold */
                }
                
                /* Enhanced signal table styles */
                .signals-table th {
                    font-weight: 600;
                    color: #ffffff;
                    background-color: #283442;
                    padding: 15px 12px;
                }
                
                .signals-table td {
                    padding: 15px 12px;
                    border-bottom: 1px solid #3d4c5e;
                }
                
                .signals-table tr:hover {
                    background-color: #2b3c4e;
                }
                
                /* Personality tooltip */
                .personality-tooltip {
                    position: relative;
                    display: inline-block;
                }
                
                .personality-tooltip:hover .tooltip-text {
                    visibility: visible;
                    opacity: 1;
                }
                
                .tooltip-text {
                    visibility: hidden;
                    opacity: 0;
                    width: 200px;
                    background-color: #283442;
                    color: #ffffff;
                    text-align: center;
                    border-radius: 6px;
                    padding: 10px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    margin-left: -100px;
                    transition: opacity 0.3s;
                    font-size: 0.9em;
                    border: 1px solid #3d4c5e;
                }
                
                /* Strategy badges */
                .strategy-badge {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    margin-right: 4px;
                    margin-bottom: 4px;
                }
                
                .strategy-value {
                    background-color: rgba(76, 175, 80, 0.2);
                    color: #4CAF50;
                }
                
                .strategy-tech {
                    background-color: rgba(33, 150, 243, 0.2);
                    color: #2196F3;
                }
                
                .strategy-trend {
                    background-color: rgba(255, 193, 7, 0.2);
                    color: #FFC107;
                }
                
                /* Emoji animations */
                @keyframes float {
                    0% { transform: translateY(0px); }
                    50% { transform: translateY(-5px); }
                    100% { transform: translateY(0px); }
                }
                
                .signal-commentary {
                    animation: float 3s ease-in-out infinite;
                }
                
                /* Action-specific styles */
                .signal-strong_buy {
                    background-color: rgba(0, 255, 0, 0.2);
                    color: #00ff00;
                    font-weight: bold;
                    border: 1px solid #00ff00;
                    padding: 4px 8px;
                    border-radius: 4px;
                    animation: pulse-green 2s infinite;
                }
                
                .signal-buy {
                    background-color: rgba(76, 175, 80, 0.2);
                    color: #4caf50;
                    font-weight: bold;
                    border: 1px solid #4caf50;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                
                .signal-scale_in {
                    background-color: rgba(139, 195, 74, 0.2);
                    color: #8bc34a;
                    font-weight: bold;
                    border: 1px solid #8bc34a;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                
                .signal-watch {
                    background-color: rgba(255, 193, 7, 0.2);
                    color: #ffc107;
                    font-weight: bold;
                    border: 1px solid #ffc107;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                
                .signal-hold {
                    background-color: rgba(158, 158, 158, 0.2);
                    color: #9e9e9e;
                    font-weight: bold;
                    border: 1px solid #9e9e9e;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                
                .signal-scale_out {
                    background-color: rgba(255, 152, 0, 0.2);
                    color: #ff9800;
                    font-weight: bold;
                    border: 1px solid #ff9800;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                
                .signal-sell {
                    background-color: rgba(244, 67, 54, 0.2);
                    color: #f44336;
                    font-weight: bold;
                    border: 1px solid #f44336;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                
                .signal-strong_sell {
                    background-color: rgba(211, 47, 47, 0.2);
                    color: #d32f2f;
                    font-weight: bold;
                    border: 1px solid #d32f2f;
                    padding: 4px 8px;
                    border-radius: 4px;
                    animation: pulse-red 2s infinite;
                }
                
                /* Animations */
                @keyframes pulse-green {
                    0% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.4); }
                    70% { box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
                }
                
                @keyframes pulse-red {
                    0% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0.4); }
                    70% { box-shadow: 0 0 0 10px rgba(211, 47, 47, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(211, 47, 47, 0); }
                }
                
                /* Signal commentary styles */
                .signal-commentary {
                    font-style: italic;
                    padding: 8px;
                    border-radius: 4px;
                    background: rgba(33, 33, 33, 0.2);
                    margin: 4px 0;
                    transition: all 0.3s ease;
                    animation: float 3s ease-in-out infinite;
                }
                
                .signal-commentary:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                }
                
                @keyframes float {
                    0% { transform: translateY(0px); }
                    50% { transform: translateY(-5px); }
                    100% { transform: translateY(0px); }
                }
                
                /* Confidence bar styles */
                .confidence-container {
                    width: 100%;
                    background: rgba(33, 33, 33, 0.2);
                    border-radius: 4px;
                    overflow: hidden;
                    position: relative;
                }
                
                .confidence-bar {
                    height: 4px;
                    background: linear-gradient(90deg, #4caf50, #ffc107, #f44336);
                    transition: width 0.3s ease;
                }
                
                /* Agent personality badges */
                .agent-warren {
                    background: rgba(76, 175, 80, 0.2);
                    color: #4caf50;
                }
                
                .agent-elon {
                    background: rgba(33, 150, 243, 0.2);
                    color: #2196f3;
                }
                
                .agent-technical {
                    background: rgba(156, 39, 176, 0.2);
                    color: #9c27b0;
                }
                
                /* Trade status styles */
                .trade-status-executed {
                    color: #4CAF50;
                    font-weight: bold;
                }
                .trade-status-pending {
                    color: #FF9800;
                    font-weight: bold;
                }
                .trade-status-no-action {
                    color: #9E9E9E;
                    font-weight: bold;
                }
                
                /* Signals table improvements */
                .signals-table-container {
                    overflow-x: auto;
                    margin-top: 15px;
                    border-radius: 12px;
                    background-color: #1E1E1E;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                }
                
                .signals-table-content {
                    width: 100%;
                    border-collapse: collapse;
                }
                
                .signals-table-content th {
                    background-color: #333333;
                    padding: 15px;
                    text-align: left;
                    font-weight: 500;
                    color: #ffffff;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                
                .signals-table-content td {
                    padding: 12px 15px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .signals-table-content tr:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                
                /* Market overview improvements */
                .market-overview-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }
                
                .market-overview-table th {
                    background-color: #333333;
                    padding: 15px;
                    text-align: left;
                    font-weight: 500;
                    color: #ffffff;
                }
                
                .market-overview-table td {
                    padding: 12px 15px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .market-overview-table tr:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                
                /* Progress bar improvements */
                .progress-bar-container {
                    width: 100%;
                    background-color: #333333;
                    border-radius: 8px;
                    position: relative;
                    height: 20px;
                    overflow: hidden;
                    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
                }
                
                .progress-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #4CAF50, #8BC34A);
                    border-radius: 8px;
                    transition: width 0.5s ease-in-out;
                }
                
                .progress-text {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: white;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
                }
                
                /* Memory status improvements */
                .memory-status {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: linear-gradient(135deg, #2196F3, #0D47A1);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    font-size: 0.9em;
                    z-index: 1000;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    transition: all 0.3s ease;
                }
                
                .memory-status .icon {
                    font-size: 1.3em;
                }
                
                .memory-status.saved {
                    background: linear-gradient(135deg, #4CAF50, #2E7D32);
                    animation: fadeOut 3s forwards;
                    animation-delay: 2s;
                }
                
                @keyframes fadeOut {
                    from { opacity: 1; transform: translateY(0); }
                    to { opacity: 0; transform: translateY(20px); visibility: hidden; }
                }
                
                /* Dropdown improvements */
                .dropdown {
                    margin-bottom: 15px;
                    width: 100%;
                }
                
                /* Coin pair dropdown */
                .coin-pair-dropdown .Select-value {
                    background-color: #673AB7;
                    border-color: #512DA8;
                }
                
                .coin-pair-dropdown .Select-value-label {
                    color: white !important;
                }
                
                /* Trader quotes */
                .trader-quote {
                    font-style: italic;
                    color: #a8b2c1;
                    font-size: 0.9em;
                    max-width: 200px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }
                
                /* Meme coin styling */
                .meme-coin {
                    position: relative;
                    display: inline-block;
                }
                
                .meme-coin::after {
                    content: "🚀";
                    position: absolute;
                    top: -5px;
                    right: -10px;
                    font-size: 0.8em;
                }
                
                /* Alt coin styling */
                .alt-coin {
                    position: relative;
                    display: inline-block;
                }
                
                .alt-coin::after {
                    content: "⭐";
                    position: absolute;
                    top: -5px;
                    right: -10px;
                    font-size: 0.8em;
                }
                
                /* Coin categories */
                .coin-category {
                    display: inline-block;
                    font-size: 0.7em;
                    padding: 2px 6px;
                    border-radius: 10px;
                    margin-left: 5px;
                }
                
                .coin-category.major {
                    background-color: #4CAF50;
                    color: white;
                }
                
                .coin-category.meme {
                    background-color: #FF9800;
                    color: white;
                }
                
                .coin-category.alt {
                    background-color: #2196F3;
                    color: white;
                }
                
                .coin-category.ratio {
                    background-color: #673AB7;
                    color: white;
                }
                
                /* Trade history styles */
                .trade-history-section {
                    margin-top: 30px;
                    padding: 20px;
                    background-color: #2b3c4e;
                    border-radius: 10px;
                }
                
                .trade-history-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-top: 20px;
                }
                
                .trade-history-card {
                    flex: 1;
                    min-width: 300px;
                    background-color: #1f2630;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                
                .trade-count {
                    color: #a8b2c1;
                    margin-bottom: 15px;
                    font-size: 0.9em;
                }
                
                .trade-list {
                    max-height: 400px;
                    overflow-y: auto;
                    padding-right: 5px;
                }
                
                .trade-item {
                    margin-bottom: 15px;
                    padding: 12px;
                    border-radius: 8px;
                    background-color: #283442;
                    border-left: 4px solid;
                }
                
                .trade-item.trade-buy {
                    border-color: #4CAF50;
                }
                
                .trade-item.trade-sell {
                    border-color: #F44336;
                }
                
                .trade-header {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .trade-action {
                    font-weight: bold;
                    padding: 2px 8px;
                    border-radius: 4px;
                }
                
                .trade-action.trade-buy {
                    background-color: rgba(76, 175, 80, 0.2);
                    color: #4CAF50;
                }
                
                .trade-action.trade-sell {
                    background-color: rgba(244, 67, 54, 0.2);
                    color: #F44336;
                }
                
                .trade-symbol {
                    color: #a8b2c1;
                }
                
                .trade-details {
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }
                
                .trade-detail {
                    display: flex;
                    justify-content: space-between;
                }
                
                .trade-label {
                    color: #a8b2c1;
                    font-size: 0.9em;
                }
                
                .trade-value {
                    font-weight: 500;
                }
                
                .trade-time {
                    font-size: 0.85em;
                    color: #a8b2c1;
                }
                
                .no-trades-message {
                    padding: 20px;
                    text-align: center;
                    color: #a8b2c1;
                    font-style: italic;
                }
                
                /* Modern UI improvements */
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    background-color: #121212;
                    color: #ffffff;
                    line-height: 1.6;
                }
                
                .container {
                    padding: 20px;
                    max-width: 1800px;
                    margin: 0 auto;
                }
                
                .header {
                    text-align: center;
                    padding: 30px 20px;
                    background: linear-gradient(135deg, #1a237e, #0d47a1);
                    border-radius: 15px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
                    position: relative;
                    overflow: hidden;
                }
                
                .dashboard-title {
                    margin: 0;
                    color: #ffffff;
                    font-size: 2.8em;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                    letter-spacing: 1px;
                }
                
                .dashboard-subtitle {
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 1.2em;
                    margin-top: 10px;
                    font-weight: 300;
                }
                
                /* Navigation improvements */
                .nav-container {
                    display: flex;
                    justify-content: center;
                    margin-bottom: 30px;
                    gap: 15px;
                }
                
                .nav-button {
                    background-color: #252525;
                    color: #ffffff;
                    border: none;
                    padding: 12px 25px;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-size: 1.1em;
                    font-weight: 500;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    position: relative;
                    overflow: hidden;
                }
                
                .nav-button::after {
                    content: '';
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    height: 3px;
                    background-color: #2196F3;
                    transform: scaleX(0);
                    transition: transform 0.3s ease;
                }
                
                .nav-button:hover {
                    background-color: #333333;
                    transform: translateY(-2px);
                }
                
                .nav-button.active {
                    background-color: #333333;
                    color: #2196F3;
                }
                
                .nav-button.active::after {
                    transform: scaleX(1);
                }
                
                /* Main content improvements */
                .main-content {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                .left-panel, .right-panel {
                    background-color: #252525;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                }
                
                .control-panel, .chart-panel, .signals-panel {
                    background-color: #1E1E1E;
                    padding: 25px;
                    border-radius: 12px;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }
                
                .control-panel:hover, .chart-panel:hover, .signals-panel:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                }
                
                .panel-title {
                    margin-top: 0;
                    margin-bottom: 25px;
                    color: #ffffff;
                    font-size: 1.5em;
                    font-weight: 500;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    padding-bottom: 15px;
                    position: relative;
                }
                
                .panel-title::after {
                    content: '';
                    position: absolute;
                    bottom: -1px;
                    left: 0;
                    width: 50px;
                    height: 3px;
                    background-color: #2196F3;
                }
                
                /* Chart grid improvements */
                .chart-grid {
                    display: grid;
                    gap: 20px;
                    margin-bottom: 25px;
                }
                
                .chart-grid-2x2 {
                    grid-template-columns: repeat(2, 1fr);
                    grid-template-rows: repeat(2, 400px);
                }
                
                .chart-grid-1x4 {
                    grid-template-columns: 1fr;
                    grid-template-rows: repeat(4, 300px);
                }
                
                .chart-grid-2x3 {
                    grid-template-columns: repeat(2, 1fr);
                    grid-template-rows: repeat(3, 300px);
                }
                
                .chart-container-2x2, .chart-container-1x4, .chart-container-2x3 {
                    background-color: #1E1E1E;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease;
                }
                
                .chart-container-2x2:hover, .chart-container-1x4:hover, .chart-container-2x3:hover {
                    transform: scale(1.02);
                }
                
                /* Form controls improvements */
                label {
                    display: block;
                    margin-bottom: 8px;
                    color: #B0B0B0;
                    font-weight: 500;
                }
                
                .dropdown {
                    margin-bottom: 20px;
                }
                
                /* Holdings summary styles */
                .holdings-summary-section {
                    margin-bottom: 30px;
                }
                
                .holdings-title {
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: #ffffff;
                    font-size: 1.5em;
                    text-align: center;
                }
                
                .holdings-summary-container {
                    padding: 10px;
                }
                
                .holdings-summary-grid {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }
                
                .holdings-summary-card {
                    flex: 1;
                    min-width: 280px;
                    max-width: 350px;
                    background: linear-gradient(145deg, #1E1E1E, #252525);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    transition: all 0.3s ease;
                    border-left: 5px solid #2196F3;
                    position: relative;
                    overflow: hidden;
                }
                
                .holdings-summary-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3);
                }
                
                .holdings-summary-card.leading {
                    border-left: 5px solid #4CAF50;
                    background: linear-gradient(145deg, #1E1E1E, #2E3B2E);
                }
                
                .holdings-summary-card.leading::after {
                    content: '🏆';
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    font-size: 1.5em;
                    opacity: 0.8;
                }
                
                .summary-header {
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                .value-container {
                    text-align: center;
                    margin-bottom: 20px;
                }
                
                .total-value {
                    font-size: 2.2em;
                    font-weight: bold;
                    color: #ffffff;
                    margin-bottom: 5px;
                }
                
                .value-label {
                    color: #B0B0B0;
                    font-size: 0.9em;
                }
                
                .balance-container {
                    margin-bottom: 20px;
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                    padding: 15px;
                }
                
                .balance-row {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                }
                
                .balance-row:last-child {
                    margin-bottom: 0;
                }
                
                .balance-label {
                    color: #B0B0B0;
                }
                
                .balance-value {
                    font-weight: 500;
                }
                
                .goal-container {
                    text-align: center;
                }
                
                .goal-progress-bar {
                    height: 10px;
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 5px;
                    overflow: hidden;
                    margin-bottom: 8px;
                }
                
                .goal-progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #2196F3, #4CAF50);
                    border-radius: 5px;
                    transition: width 0.5s ease;
                }
                
                .goal-status {
                    font-size: 0.9em;
                    color: #B0B0B0;
                }
                
                /* Detailed holdings styles */
                .holdings-grid {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }
                
                .holdings-card {
                    flex: 1;
                    min-width: 300px;
                    background-color: #1E1E1E;
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease;
                }
                
                .holdings-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                }
                
                .holdings-card-title {
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: #ffffff;
                    font-size: 1.3em;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    padding-bottom: 10px;
                }
                
                .holdings-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 8px;
                }
                
                .usdt-balance {
                    background-color: rgba(33, 150, 243, 0.1);
                    border-left: 3px solid #2196F3;
                }
                
                .crypto-total {
                    background-color: rgba(76, 175, 80, 0.1);
                    border-left: 3px solid #4CAF50;
                    margin-bottom: 20px;
                }
                
                .holdings-label {
                    color: #B0B0B0;
                    font-weight: 500;
                }
                
                .holdings-value {
                    font-weight: bold;
                }
                
                .holdings-detail {
                    margin-top: 15px;
                    max-height: 300px;
                    overflow-y: auto;
                }
                
                .holdings-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                
                .holdings-table th {
                    background-color: #252525;
                    padding: 10px;
                    text-align: left;
                    font-weight: 500;
                    color: #B0B0B0;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                
                .holdings-table td {
                    padding: 10px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }
                
                .holdings-table tr:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                
                .coin-category-icon {
                    margin-right: 8px;
                    font-size: 1.1em;
                }
                
                .major-coin {
                    color: #2196F3;
                }
                
                .meme-coin {
                    color: #FF9800;
                }
                
                .alt-coin {
                    color: #9C27B0;
                }
                
                /* Coin tags styles */
                .coin-section {
                    margin-bottom: 20px;
                }
                
                .section-label {
                    display: block;
                    margin-bottom: 10px;
                    color: #ffffff;
                    font-weight: 500;
                    font-size: 1.1em;
                }
                
                .coin-tags-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin-bottom: 15px;
                }
                
                .coin-tag {
                    display: inline-block;
                    padding: 6px 12px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    transition: all 0.3s ease;
                }
                
                .coin-tag:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }
                
                .usdt-pair {
                    background: linear-gradient(135deg, #2196F3, #0D47A1);
                    color: white;
                }
                
                .coin-pair {
                    background: linear-gradient(135deg, #673AB7, #4A148C);
                    color: white;
                }
                
                /* Additional chart grid layouts */
                .chart-grid-3x3 {
                    grid-template-columns: repeat(3, 1fr);
                    grid-template-rows: repeat(3, 300px);
                }
                
                .chart-grid-4x4 {
                    grid-template-columns: repeat(4, 1fr);
                    grid-template-rows: repeat(4, 250px);
                }
                
                .chart-container-3x3, .chart-container-4x4 {
                    background-color: #1E1E1E;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                    transition: transform 0.3s ease;
                }
                
                .chart-container-3x3:hover, .chart-container-4x4:hover {
                    transform: scale(1.02);
                }
                
                /* Percentage bar styles */
                .percentage-bar-container {
                    width: 100%;
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    position: relative;
                    height: 16px;
                    overflow: hidden;
                }
                
                .percentage-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #2196F3, #03A9F4);
                    border-radius: 4px;
                    transition: width 0.5s ease;
                }
                
                .percentage-text {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: white;
                    font-size: 0.8em;
                    font-weight: bold;
                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
                }
                
                /* Enhanced holdings table */
                .holdings-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                
                .holdings-table th {
                    background-color: #252525;
                    padding: 12px 10px;
                    text-align: left;
                    font-weight: 500;
                    color: #B0B0B0;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }
                
                .holdings-table td {
                    padding: 12px 10px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                    vertical-align: middle;
                }
                
                .holdings-table tr:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                
                /* Wallet composition styles */
                .wallet-composition {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                }
                
                .composition-title {
                    font-weight: 500;
                    margin-bottom: 10px;
                    color: #B0B0B0;
                    text-align: center;
                }
                
                .composition-bar {
                    height: 24px;
                    border-radius: 12px;
                    overflow: hidden;
                    display: flex;
                    margin-bottom: 10px;
                    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
                }
                
                .usdt-portion {
                    height: 100%;
                    background: linear-gradient(90deg, #2196F3, #03A9F4);
                }
                
                .crypto-portion {
                    height: 100%;
                    background: linear-gradient(90deg, #4CAF50, #8BC34A);
                }
                
                .composition-legend {
                    display: flex;
                    justify-content: space-around;
                    margin-top: 10px;
                }
                
                .legend-item {
                    display: flex;
                    align-items: center;
                    font-size: 0.9em;
                }
                
                .usdt-dot, .crypto-dot {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 6px;
                }
                
                .usdt-dot {
                    background-color: #2196F3;
                }
                
                .crypto-dot {
                    background-color: #4CAF50;
                }
                
                /* Enhanced holdings card styles */
                .holdings-card-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                
                .holdings-total-value {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #4CAF50;
                }
                
                .holdings-subtitle {
                    margin-top: 20px;
                    margin-bottom: 15px;
                    color: #B0B0B0;
                    font-size: 1.1em;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    padding-bottom: 8px;
                }
                
                .wallet-summary {
                    background-color: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                
                .wallet-bar {
                    height: 24px;
                    border-radius: 12px;
                    overflow: hidden;
                    display: flex;
                    margin-bottom: 15px;
                    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
                }
                
                .usdt-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #2196F3, #03A9F4);
                }
                
                .crypto-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #4CAF50, #8BC34A);
                }
                
                .wallet-legend {
                    display: flex;
                    justify-content: space-around;
                }
                
                .legend-row {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .legend-label {
                    color: #B0B0B0;
                    font-size: 0.9em;
                }
                
                .legend-value {
                    font-weight: bold;
                }
                
                /* Table cell styles */
                .asset-cell {
                    font-weight: 500;
                }
                
                .amount-cell {
                    font-family: monospace;
                    text-align: right;
                }
                
                .price-cell, .value-cell {
                    text-align: right;
                    font-weight: 500;
                }
                
                .percentage-cell {
                    width: 150px;
                }
                
                .holding-row.even {
                    background-color: rgba(255, 255, 255, 0.02);
                }
                
                .holding-row:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                }
                
                .no-holdings-message {
                    text-align: center;
                    color: #B0B0B0;
                    font-style: italic;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    @app.callback(
        Output('memory-status', 'className'),
        [Input('memory-save-interval', 'n_intervals')]
    )
    def update_memory_status(n):
        """Update the memory status indicator when state is saved."""
        if n > 0:
            # Save the state
            trading_system._save_state()
            return "memory-status saved"
        return "memory-status"
    
    return app

def main():
    # Initialize trading system
    trading_system = TradingSystem()
    
    # Create and start the dashboard
    app = create_dashboard(trading_system)
    
    # Run the trading system in a separate thread
    trading_thread = Thread(target=trading_system.run)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Run the dashboard
    app.run_server(debug=True, use_reloader=False)

if __name__ == "__main__":
    main() 