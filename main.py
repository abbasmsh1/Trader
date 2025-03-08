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
        
        # Add properly structured wallet components with correct IDs
        html.Div([
            # Wallet Overview Section
            html.Div([
                html.H2("Agent Wallets Overview", className='section-title'),
                html.Div(id='wallet-overview-cards', className='wallet-overview-grid')
            ], className='wallet-section'),
            
            # Portfolio Performance Section
            html.Div([
                html.H2("Portfolio Performance", className='section-title'),
                dcc.Dropdown(
                    id='agent-selector',
                    options=[],  # Will be populated when needed
                    className='agent-dropdown'
                ),
                html.Div(id='portfolio-performance', className='performance-container')
            ], className='portfolio-section'),
            
            # Trade History Section
            html.Div([
                html.H2("Trade History", className='section-title'),
                html.Div(id='trade-history-table', className='trade-history-container')
            ], className='trade-history-section')
        ], id='wallet-view-container', style={'display': 'none'}),
        
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
        Output('agent-selector', 'options'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_agent_selector(n):
        """Update the agent selector dropdown options."""
        return [{'label': agent.name, 'value': i} for i, agent in enumerate(trading_system.agents)]

    @app.callback(
        [Output('wallet-overview-cards', 'children'),
         Output('portfolio-performance', 'children'),
         Output('trade-history-table', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('agent-selector', 'value')]
    )
    def update_wallet_view(n, selected_agent_idx):
        """Update the wallet view components."""
        # Initialize empty containers for the components
        wallet_cards = []
        performance_charts = []
        trade_history = html.Div("No trade history available")

        try:
            # Get current prices for all symbols
            current_prices = {}
            for symbol in trading_system.symbols:
                df = trading_system.data_fetcher.get_historical_data(symbol)
                if not df.empty:
                    current_prices[symbol] = float(df['close'].iloc[-1])
            
            # Create wallet overview cards
            for idx, agent in enumerate(trading_system.agents):
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
                                html.Span(f"${metrics['usdt_balance']:.2f}", className='metric-value')
                            ], className='metric-row'),
                            html.Div([
                                html.Span("Holdings Value:", className='metric-label'),
                                html.Span(f"${metrics['holdings_value_usdt']:.2f}", className='metric-value')
                            ], className='metric-row')
                        ], className='wallet-card-content')
                    ], className=f'wallet-card {"selected" if idx == selected_agent_idx else ""}')
                )
            
            # If an agent is selected, create performance charts and trade history
            if selected_agent_idx is not None:
                selected_agent = trading_system.agents[selected_agent_idx]
                
                # Create portfolio value chart
                portfolio_value_fig = go.Figure()
                portfolio_value_fig.add_trace(go.Scatter(
                    x=[trade['timestamp'] for trade in selected_agent.trade_history],
                    y=[trade['portfolio_value'] for trade in selected_agent.trade_history],
                    mode='lines',
                    name='Portfolio Value'
                ))
                portfolio_value_fig.update_layout(
                    title='Portfolio Value Over Time',
                    template='plotly_dark',
                    height=400
                )
                
                # Create holdings distribution chart
                holdings = selected_agent.get_holdings()
                holdings_fig = go.Figure(data=[go.Pie(
                    labels=list(holdings.keys()),
                    values=list(holdings.values()),
                    hole=.3
                )])
                holdings_fig.update_layout(
                    title='Holdings Distribution',
                    template='plotly_dark',
                    height=400
                )
                
                performance_charts = [
                    dcc.Graph(figure=portfolio_value_fig),
                    dcc.Graph(figure=holdings_fig)
                ]
                
                # Create trade history table
                trade_history = html.Table([
                    html.Thead(html.Tr([
                        html.Th("Time"),
                        html.Th("Symbol"),
                        html.Th("Action"),
                        html.Th("Amount"),
                        html.Th("Price")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(datetime.fromtimestamp(trade['timestamp']).strftime('%Y-%m-%d %H:%M')),
                            html.Td(trade['symbol']),
                            html.Td(trade['action']),
                            html.Td(f"{trade['amount']:.4f}"),
                            html.Td(f"${trade['price']:.2f}")
                        ]) for trade in reversed(selected_agent.trade_history[-10:])  # Show last 10 trades
                    ])
                ], className='trade-history-table')
        
        except Exception as e:
            print(f"Error updating wallet view: {str(e)}")
            return [], [], html.Div(f"Error: {str(e)}")
        
        return wallet_cards, performance_charts, trade_history
    
    # Add CSS styles
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* Global Styles */
                :root {
                    --primary-color: #2196F3;
                    --secondary-color: #f50057;
                    --success-color: #4CAF50;
                    --warning-color: #ff9800;
                    --error-color: #f44336;
                    --background-dark: #1a1a1a;
                    --background-card: #2a2a2a;
                    --background-hover: #303030;
                    --text-primary: #ffffff;
                    --text-secondary: #9e9e9e;
                    --border-color: #404040;
                }

                body {
                    background-color: var(--background-dark);
                    color: var(--text-primary);
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                    margin: 0;
                    padding: 0;
                }

                /* Dashboard Layout */
                .container {
                    max-width: 1800px;
                    margin: 0 auto;
                    padding: 1rem;
                }

                .header {
                    margin-bottom: 2rem;
                    padding: 1rem;
                    background: var(--background-card);
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }

                /* News Section Styles */
                .news-section {
                    margin: 2rem 0;
                    padding: 1rem;
                    background: var(--background-card);
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }

                .news-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }

                .news-card {
                    background: var(--background-hover);
                    border-radius: 8px;
                    padding: 1rem;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }

                .news-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                }

                .news-source {
                    color: var(--primary-color);
                    font-size: 0.9rem;
                    margin-bottom: 0.5rem;
                }

                .news-title {
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                    color: var(--text-primary);
                }

                .news-body {
                    color: var(--text-secondary);
                    font-size: 0.9rem;
                    margin-bottom: 1rem;
                    line-height: 1.4;
                }

                .news-metadata {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.8rem;
                    color: var(--text-secondary);
                }

                .sentiment-badge {
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-weight: 500;
                }

                .sentiment-positive {
                    background: rgba(76, 175, 80, 0.2);
                    color: #81c784;
                }

                .sentiment-negative {
                    background: rgba(244, 67, 54, 0.2);
                    color: #e57373;
                }

                .sentiment-neutral {
                    background: rgba(158, 158, 158, 0.2);
                    color: #bdbdbd;
                }

                /* Wallet Section Styles */
                .wallet-section {
                    margin-bottom: 2rem;
                }
                
                .wallet-overview-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }
                
                .wallet-card {
                    background: var(--background-card);
                    border-radius: 8px;
                    padding: 1rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                }
                
                .wallet-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                }
                
                .wallet-card.selected {
                    border: 2px solid var(--primary-color);
                    background: var(--background-hover);
                }
                
                .wallet-card-title {
                    color: var(--text-primary);
                    margin: 0 0 1rem 0;
                    font-size: 1.2rem;
                }
                
                .metric-row {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 0.5rem;
                }
                
                .metric-label {
                    color: var(--text-secondary);
                }
                
                .metric-value {
                    color: var(--text-primary);
                    font-weight: bold;
                }
                
                .metric-value.positive {
                    color: var(--success-color);
                }
                
                .metric-value.negative {
                    color: var(--error-color);
                }

                /* Chart Styles */
                .chart-container {
                    background: var(--background-card);
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                }

                /* Table Styles */
                .trade-history-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 1rem;
                    background: var(--background-card);
                    border-radius: 8px;
                    overflow: hidden;
                }
                
                .trade-history-table th,
                .trade-history-table td {
                    padding: 0.75rem;
                    text-align: left;
                    border-bottom: 1px solid var(--border-color);
                }
                
                .trade-history-table th {
                    background: var(--background-hover);
                    color: var(--text-primary);
                    font-weight: bold;
                }
                
                .trade-history-table tr:hover {
                    background: var(--background-hover);
                }

                /* Control Elements */
                .agent-dropdown {
                    margin: 1rem 0;
                    width: 100%;
                    max-width: 300px;
                }

                .agent-dropdown .Select-control {
                    background: var(--background-card);
                    border-color: var(--border-color);
                    color: var(--text-primary);
                }

                .agent-dropdown .Select-menu-outer {
                    background: var(--background-card);
                    border-color: var(--border-color);
                }

                .agent-dropdown .Select-option {
                    background: var(--background-card);
                    color: var(--text-primary);
                }

                .agent-dropdown .Select-option:hover {
                    background: var(--background-hover);
                }

                /* Performance Container */
                .performance-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 1rem;
                    margin-top: 1rem;
                }

                /* Responsive Design */
                @media (max-width: 768px) {
                    .container {
                        padding: 0.5rem;
                    }

                    .news-grid,
                    .wallet-overview-grid,
                    .performance-container {
                        grid-template-columns: 1fr;
                    }

                    .news-card,
                    .wallet-card {
                        margin-bottom: 1rem;
                    }
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