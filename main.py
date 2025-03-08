import os
from typing import List, Dict
import pandas as pd
from datetime import datetime
import time
from data.market_data import MarketDataFetcher
from agents.value_investor import ValueInvestor
from agents.tech_disruptor import TechDisruptor
from agents.trend_follower import TrendFollower
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from threading import Thread
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
        self.symbols = symbols or DEFAULT_SYMBOLS
        
        # Initialize trading agents with different personalities
        self.agents = [
            ValueInvestor(name="Warren Buffett AI", timeframe='1d'),
            TechDisruptor(name="Elon Musk AI", timeframe='1h'),
            TrendFollower(name="Technical Trader", risk_tolerance=0.5, timeframe='4h')
        ]
        
        self.signals_history = []
        
    def analyze_market(self, symbol: str) -> List[Dict]:
        """Analyze market data using all agents."""
        market_data = self.data_fetcher.get_historical_data(symbol)
        if market_data.empty:
            return []
        
        # Set the symbol name in the market data
        market_data.name = symbol
        
        # Get current price
        current_price = float(market_data['close'].iloc[-1])
        
        signals = []
        current_prices = {symbol: current_price}
        
        for agent in self.agents:
            # Get analysis and signal from each agent
            analysis = agent.analyze_market(market_data)
            signal = agent.generate_signal(analysis)
            
            # Execute trade based on signal
            trade_executed = agent.execute_trade(symbol, signal, current_price)
            
            # Get wallet metrics
            wallet_metrics = agent.get_wallet_metrics(current_prices)
            
            # Add agent personality info to signal
            personality_traits = agent.get_personality_traits()
            signals.append({
                'agent': agent.name,
                'personality': personality_traits['personality'],
                'symbol': symbol,
                'signal': signal,
                'risk_tolerance': agent.risk_tolerance,
                'strategy': personality_traits['strategy_preferences'],
                'market_view': personality_traits['market_beliefs'],
                'wallet_metrics': wallet_metrics,
                'trade_executed': trade_executed
            })
            
        return signals
    
    def run(self, interval: int = SYSTEM_PARAMS['update_interval']):
        """
        Run the trading system continuously.
        
        Args:
            interval (int): Update interval in seconds
        """
        while True:
            for symbol in self.symbols:
                signals = self.analyze_market(symbol)
                self.signals_history.extend(signals)
                
                # Keep only last 1000 signals
                if len(self.signals_history) > SYSTEM_PARAMS['max_signals_history']:
                    self.signals_history = self.signals_history[-SYSTEM_PARAMS['max_signals_history']:]
                
                # Print signals
                print(f"\nNew signals for {symbol} at {datetime.now()}:")
                for signal in signals:
                    print(f"{signal['agent']}: {signal['signal']['action'].upper()} "
                          f"(Confidence: {signal['signal']['confidence']:.2f})")
                
            time.sleep(interval)

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
        # Header with navigation
        html.Div([
            html.H1("AI Crypto Trading Dashboard", className='dashboard-title'),
            html.P("Real-time cryptocurrency trading signals powered by AI", className='dashboard-subtitle'),
            # Add navigation tabs
            dcc.Tabs(id='navigation-tabs', value='trading-view', children=[
                dcc.Tab(label='Trading View', value='trading-view'),
                dcc.Tab(label='Wallet & Portfolio', value='wallet-view')
            ], className='nav-tabs')
        ], className='header'),
        
        # Content container that switches between views
        html.Div(id='page-content'),
        
        # Hidden div for storing the current view state
        html.Div(id='current-view', style={'display': 'none'}),
        
        # Add placeholder divs for wallet components
        html.Div(id='wallet-overview-cards', style={'display': 'none'}),
        html.Div(id='portfolio-performance', style={'display': 'none'}),
        html.Div(id='trade-history-table', style={'display': 'none'}),
        
        # Add the agent selector dropdown to the initial layout
        dcc.Dropdown(id='agent-selector', style={'display': 'none'}),
        
        # Refresh interval
        dcc.Interval(
            id='interval-component',
            interval=SYSTEM_PARAMS['dashboard_refresh_rate'],
            n_intervals=0
        )
    ], className='container')
    
    def create_trading_view():
        """Create the main trading view layout."""
        return html.Div([
            # Main content
            html.Div([
                # Left panel - Controls
                html.Div([
                    html.Div([
                        html.H3("Trading Controls", className='panel-title'),
                        html.Label("Select Cryptocurrencies"),
                        dcc.Dropdown(
                            id='symbol-multi-dropdown',
                            options=[{'label': s.replace('/USDT', ''), 'value': s} 
                                    for s in trading_system.symbols],
                            value=trading_system.symbols[:4],
                            multi=True,
                            className='dropdown'
                        ),
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
                                options=[{'label': '', 'value': 'enabled'}],
                                value=['enabled'],
                                className='auto-refresh-toggle'
                            )
                        ], className='switch-container'),
                        html.Div([
                            html.Label("Chart Layout"),
                            dcc.RadioItems(
                                id='layout-radio',
                                options=[
                                    {'label': '2x2 Grid', 'value': '2x2'},
                                    {'label': '1x4 Row', 'value': '1x4'},
                                    {'label': '2x3 Grid', 'value': '2x3'}
                                ],
                                value='2x2',
                                className='layout-radio'
                            )
                        ], className='layout-container'),
                        html.Div([
                            html.Label("Refresh Interval"),
                            dcc.Slider(
                                id='refresh-interval-slider',
                                min=5,
                                max=60,
                                step=5,
                                value=30,
                                marks={i: f'{i}s' for i in range(5, 61, 5)},
                                className='refresh-slider'
                            )
                        ], className='slider-container')
                    ], className='control-panel'),
                    
                    # Performance Metrics Panel
                    html.Div([
                        html.H3("Performance Metrics", className='panel-title'),
                        html.Div(id='performance-metrics', className='metrics-container')
                    ], className='metrics-panel')
                ], className='left-panel'),
                
                # Right panel - Charts and Data
                html.Div([
                    # Multi-Chart Container
                    html.Div(id='multi-chart-container', className='chart-grid'),
                    
                    # Market Overview
                    html.Div([
                        html.H3("Market Overview", className='panel-title'),
                        html.Div(id='market-overview', className='market-overview')
                    ], className='market-panel'),
                    
                    # Trading Signals
                    html.Div([
                        html.H3("Latest Trading Signals", className='panel-title'),
                        html.Div(id='signals-table', className='signals-table')
                    ], className='signals-panel')
                ], className='right-panel')
            ], className='main-content')
        ])
    
    def create_wallet_view():
        """Create the wallet and portfolio view layout."""
        return html.Div([
            # Wallet Overview Section
            html.Div([
                html.H2("Agent Wallets Overview", className='section-title'),
                html.Div(id='wallet-overview-cards', className='wallet-overview-grid')
            ], className='wallet-section'),
            
            # Portfolio Performance Section
            html.Div([
                html.H2("Portfolio Performance", className='section-title'),
                # Agent selector dropdown
                dcc.Dropdown(
                    id='agent-selector',
                    options=[{'label': agent.name, 'value': i} 
                            for i, agent in enumerate(trading_system.agents)],
                    value=0,
                    className='agent-dropdown'
                ),
                # Performance charts container
                html.Div(id='portfolio-performance', className='performance-container')
            ], className='portfolio-section'),
            
            # Trade History Section
            html.Div([
                html.H2("Trade History", className='section-title'),
                html.Div(id='trade-history-table', className='trade-history-container')
            ], className='trade-history-section')
        ], className='wallet-view')
    
    @app.callback(
        Output('page-content', 'children'),
        [Input('navigation-tabs', 'value')]
    )
    def render_content(tab):
        """Render the appropriate view based on selected tab."""
        if tab == 'wallet-view':
            return create_wallet_view()
        return create_trading_view()
    
    @app.callback(
        [Output('wallet-overview-cards', 'children'),
         Output('portfolio-performance', 'children'),
         Output('trade-history-table', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('agent-selector', 'value')]
    )
    def update_wallet_view(n, selected_agent_idx):
        """Update the wallet view components."""
        # Get current prices for all symbols
        current_prices = {}
        for symbol in trading_system.symbols:
            df = trading_system.data_fetcher.get_historical_data(symbol)
            if not df.empty:
                current_prices[symbol] = float(df['close'].iloc[-1])
        
        # Create wallet overview cards
        wallet_cards = []
        for agent in trading_system.agents:
            metrics = agent.get_wallet_metrics(current_prices)
            wallet_cards.append(
                html.Div([
                    html.H3(agent.name, className='wallet-card-title'),
                    html.Div([
                        html.Div([
                            html.Span("Total Value:", className='metric-label'),
                            html.Span(f"${metrics['total_value_usdt']:.2f}", className='metric-value')
                        ], className='metric-row'),
                        html.Div([
                            html.Span("USDT Balance:", className='metric-label'),
                            html.Span(f"${metrics['balance_usdt']:.2f}", className='metric-value')
                        ], className='metric-row'),
                        html.Div([
                            html.Span("Total Return:", className='metric-label'),
                            html.Span(
                                f"{metrics['total_return_pct']:+.2f}%",
                                className=f"metric-value {'positive' if metrics['total_return_pct'] > 0 else 'negative'}"
                            )
                        ], className='metric-row'),
                        html.Div([
                            html.Span("Holdings:", className='metric-label'),
                            html.Div([
                                html.Div(
                                    f"{amount:.4f} {sym} (${amount * current_prices.get(sym, 0):.2f})"
                                ) for sym, amount in metrics['holdings'].items()
                            ], className='holdings-list')
                        ], className='metric-row')
                    ], className='wallet-card-content')
                ], className='wallet-card')
            )
        
        # Create performance charts for selected agent
        selected_agent = trading_system.agents[selected_agent_idx]
        performance_charts = [
            # Portfolio Value Chart
            dcc.Graph(
                figure=create_portfolio_value_chart(selected_agent, current_prices),
                className='performance-chart'
            ),
            # Holdings Distribution Chart
            dcc.Graph(
                figure=create_holdings_distribution_chart(selected_agent, current_prices),
                className='performance-chart'
            )
        ]
        
        # Create trade history table
        trade_history = create_trade_history_table(selected_agent)
        
        return wallet_cards, performance_charts, trade_history
    
    def create_portfolio_value_chart(agent, current_prices):
        """Create a line chart showing portfolio value over time."""
        # Get trade history
        trades_df = agent.wallet.get_trade_history_df()
        if trades_df.empty:
            return go.Figure()
        
        # Calculate cumulative portfolio value
        trades_df['cumulative_value'] = trades_df.apply(
            lambda row: agent.wallet.get_total_value_usdt(current_prices),
            axis=1
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_value'],
            mode='lines+markers',
            name='Portfolio Value'
        ))
        
        fig.update_layout(
            title=f'{agent.name} - Portfolio Value Over Time',
            xaxis_title='Time',
            yaxis_title='Value (USDT)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_holdings_distribution_chart(agent, current_prices):
        """Create a pie chart showing current holdings distribution."""
        metrics = agent.get_wallet_metrics(current_prices)
        holdings_value = []
        labels = []
        
        # Add USDT balance
        if metrics['balance_usdt'] > 0:
            holdings_value.append(metrics['balance_usdt'])
            labels.append('USDT')
        
        # Add crypto holdings
        for symbol, amount in metrics['holdings'].items():
            if symbol in current_prices:
                value = amount * current_prices[symbol]
                holdings_value.append(value)
                labels.append(symbol.replace('/USDT', ''))
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=holdings_value,
            hole=.3
        )])
        
        fig.update_layout(
            title=f'{agent.name} - Holdings Distribution',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_trade_history_table(agent):
        """Create a table showing trade history."""
        trades_df = agent.wallet.get_trade_history_df()
        if trades_df.empty:
            return html.Div("No trades yet", className='no-trades-message')
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Time"),
                html.Th("Symbol"),
                html.Th("Action"),
                html.Th("Amount"),
                html.Th("Price"),
                html.Th("Total")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')),
                    html.Td(trade['symbol'].replace('/USDT', '')),
                    html.Td(
                        html.Span(
                            trade['action'],
                            className=f"trade-action-{trade['action'].lower()}"
                        )
                    ),
                    html.Td(f"{trade['amount_crypto']:.4f}"),
                    html.Td(f"${trade['price']:.2f}"),
                    html.Td(f"${trade['amount_usdt']:.2f}")
                ]) for _, trade in trades_df.iterrows()
            ])
        ], className='trade-history-table')
    
    # Add new CSS styles for the wallet view
    app.index_string = app.index_string.replace(
        '</style>',
        '''
        /* Wallet view styles */
        .wallet-view {
            padding: 20px;
        }
        
        .section-title {
            color: #ffffff;
            margin-bottom: 20px;
        }
        
        .wallet-overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .wallet-card {
            background-color: #2b3c4e;
            border-radius: 10px;
            padding: 20px;
            transition: transform 0.2s;
        }
        
        .wallet-card:hover {
            transform: translateY(-5px);
        }
        
        .wallet-card-title {
            color: #ffffff;
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .wallet-card-content {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .metric-label {
            color: #a8b2c1;
        }
        
        .metric-value {
            font-weight: bold;
        }
        
        .metric-value.positive {
            color: #00ff9f;
        }
        
        .metric-value.negative {
            color: #ff4757;
        }
        
        .holdings-list {
            display: flex;
            flex-direction: column;
            gap: 5px;
            margin-top: 5px;
        }
        
        .portfolio-section {
            margin-bottom: 30px;
        }
        
        .agent-dropdown {
            margin-bottom: 20px;
        }
        
        .performance-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .performance-chart {
            background-color: #2b3c4e;
            border-radius: 10px;
            padding: 15px;
        }
        
        .trade-history-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #2b3c4e;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .trade-history-table th,
        .trade-history-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #3d4c5e;
        }
        
        .trade-history-table th {
            background-color: #283442;
            color: #ffffff;
            font-weight: 600;
        }
        
        .trade-action-buy {
            color: #00ff9f;
            font-weight: bold;
        }
        
        .trade-action-sell {
            color: #ff4757;
            font-weight: bold;
        }
        
        .no-trades-message {
            text-align: center;
            padding: 20px;
            color: #a8b2c1;
        }
        
        /* Navigation tabs styles */
        .nav-tabs {
            margin-top: 20px;
        }
        
        .nav-tabs .tab {
            padding: 15px 20px;
            color: #ffffff;
            background-color: #2b3c4e;
            border: none;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }
        
        .nav-tabs .tab--selected {
            background-color: #3d4c5e;
            border-bottom: 3px solid #00ff9f;
        }
        </style>
        '''
    )
    
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