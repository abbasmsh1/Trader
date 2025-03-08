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
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
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
            html.P("Real-time cryptocurrency trading signals powered by AI", className='dashboard-subtitle')
        ], className='header'),
        
        # Navigation tabs
        dcc.Tabs(id='main-tabs', value='trading-view', className='tabs', children=[
            dcc.Tab(label='Trading View', value='trading-view', className='tab'),
            dcc.Tab(label='Traders Comparison', value='traders-comparison', className='tab')
        ]),
        
        # Content container
        html.Div(id='tab-content', className='content-container'),
        
        # Interval component for auto-refresh
        dcc.Interval(
            id='interval-component',
            interval=SYSTEM_PARAMS['dashboard_refresh_rate'],
            n_intervals=0
        )
    ], className='dashboard')
    
    # Callback to render the selected tab content
    @app.callback(
        Output('tab-content', 'children'),
        [Input('main-tabs', 'value')]
    )
    def render_tab_content(tab):
        if tab == 'trading-view':
            return create_trading_view(trading_system)
        elif tab == 'traders-comparison':
            return create_traders_comparison(trading_system)
        return html.Div("Tab content not found")
    
    def create_trading_view(trading_system):
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
    
    def create_traders_comparison(trading_system):
        """Create the traders comparison view layout."""
        return html.Div([
            html.H2("Traders Performance Comparison", className='section-title'),
            
            # Performance metrics cards
            html.Div(id='traders-performance-cards', className='performance-cards-container'),
            
            # Portfolio comparison charts
            html.Div([
                html.H3("Portfolio Value Comparison", className='subsection-title'),
                html.Div(id='portfolio-value-chart', className='comparison-chart')
            ], className='chart-section'),
            
            # Holdings comparison
            html.Div([
                html.H3("Holdings Comparison", className='subsection-title'),
                html.Div(id='holdings-comparison', className='holdings-comparison')
            ], className='chart-section'),
            
            # Trade history comparison
            html.Div([
                html.H3("Trade Activity", className='subsection-title'),
                html.Div(id='trade-activity-comparison', className='trade-activity')
            ], className='chart-section')
        ], className='traders-comparison-view')
    
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
                    df = trading_system.data_fetcher.get_historical_data(symbol)
                    if not df.empty:
                        symbol_base = symbol.split('/')[0]
                        current_prices[symbol_base] = float(df['close'].iloc[-1])
                except Exception as e:
                    print(f"Error getting price for {symbol}: {str(e)}")
            
            # Create performance cards for each agent
            performance_cards = []
            portfolio_values = []
            agent_colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
            
            for i, agent in enumerate(trading_system.agents):
                # Get wallet metrics
                metrics = agent.wallet.get_performance_metrics(current_prices)
                
                # Store portfolio value for chart
                portfolio_values.append({
                    'agent': agent.name,
                    'value': metrics['total_value_usdt'],
                    'return': metrics['total_return_pct'],
                    'color': agent_colors[i % len(agent_colors)]
                })
                
                # Create performance card
                card = html.Div([
                    html.H3(agent.name, className='card-title'),
                    html.Div([
                        html.Div([
                            html.Span("Total Value:", className='metric-label'),
                            html.Span(f"${metrics['total_value_usdt']:.2f}", className='metric-value')
                        ], className='metric-row'),
                        html.Div([
                            html.Span("Return:", className='metric-label'),
                            html.Span(
                                f"{metrics['total_return_pct']:.2f}%", 
                                className=f"metric-value {'positive' if metrics['total_return_pct'] >= 0 else 'negative'}"
                            )
                        ], className='metric-row'),
                        html.Div([
                            html.Span("USDT Balance:", className='metric-label'),
                            html.Span(f"${metrics['balance_usdt']:.2f}", className='metric-value')
                        ], className='metric-row'),
                        html.Div([
                            html.Span("Trades:", className='metric-label'),
                            html.Span(f"{metrics['trade_count']}", className='metric-value')
                        ], className='metric-row'),
                        html.Div([
                            html.Span("Strategy:", className='metric-label'),
                            html.Span(agent.personality.split(' - ')[0], className='metric-value')
                        ], className='metric-row')
                    ], className='card-content')
                ], className=f'performance-card {agent.name.lower().replace(" ", "-")}', style={'borderColor': agent_colors[i % len(agent_colors)]})
                
                performance_cards.append(card)
            
            # Create portfolio value comparison chart
            portfolio_fig = go.Figure()
            
            for pv in portfolio_values:
                portfolio_fig.add_trace(go.Bar(
                    x=[pv['agent']],
                    y=[pv['value']],
                    name=pv['agent'],
                    marker_color=pv['color'],
                    text=[f"${pv['value']:.2f}<br>{pv['return']:.2f}%"],
                    textposition='auto'
                ))
            
            portfolio_fig.update_layout(
                title='Portfolio Value by Trader',
                template='plotly_dark',
                height=400,
                yaxis_title='Value (USDT)',
                showlegend=False,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            portfolio_chart = dcc.Graph(figure=portfolio_fig)
            
            # Create holdings comparison
            holdings_cards = []
            
            for i, agent in enumerate(trading_system.agents):
                metrics = agent.wallet.get_performance_metrics(current_prices)
                holdings = metrics['holdings']
                
                if not holdings:
                    holdings_card = html.Div([
                        html.H4(agent.name, className='holdings-card-title'),
                        html.P("No holdings", className='no-holdings')
                    ], className='holdings-card')
                else:
                    holdings_items = []
                    for symbol, amount in holdings.items():
                        value = amount * current_prices.get(symbol, 0)
                        holdings_items.append(html.Div([
                            html.Span(f"{symbol}:", className='holding-symbol'),
                            html.Span(f"{amount:.6f}", className='holding-amount'),
                            html.Span(f"(${value:.2f})", className='holding-value')
                        ], className='holding-item'))
                    
                    holdings_card = html.Div([
                        html.H4(agent.name, className='holdings-card-title'),
                        html.Div(holdings_items, className='holdings-list')
                    ], className='holdings-card')
                
                holdings_cards.append(holdings_card)
            
            # Create trade activity comparison
            trade_activity_cards = []
            
            for i, agent in enumerate(trading_system.agents):
                trade_history = agent.wallet.get_trade_history_df()
                
                if trade_history.empty:
                    trade_card = html.Div([
                        html.H4(agent.name, className='trade-card-title'),
                        html.P("No trades executed", className='no-trades')
                    ], className='trade-card')
                else:
                    # Calculate trade statistics
                    buy_count = len(trade_history[trade_history['action'] == 'BUY'])
                    sell_count = len(trade_history[trade_history['action'] == 'SELL'])
                    
                    # Create trade summary
                    trade_card = html.Div([
                        html.H4(agent.name, className='trade-card-title'),
                        html.Div([
                            html.Div([
                                html.Span("Total Trades:", className='trade-stat-label'),
                                html.Span(f"{len(trade_history)}", className='trade-stat-value')
                            ], className='trade-stat'),
                            html.Div([
                                html.Span("Buys:", className='trade-stat-label'),
                                html.Span(f"{buy_count}", className='trade-stat-value buy')
                            ], className='trade-stat'),
                            html.Div([
                                html.Span("Sells:", className='trade-stat-label'),
                                html.Span(f"{sell_count}", className='trade-stat-value sell')
                            ], className='trade-stat'),
                            html.Div([
                                html.Span("Last Trade:", className='trade-stat-label'),
                                html.Span(
                                    f"{trade_history['action'].iloc[-1]} {trade_history['symbol'].iloc[-1]}", 
                                    className=f"trade-stat-value {trade_history['action'].iloc[-1].lower()}"
                                ) if not trade_history.empty else html.Span("None", className='trade-stat-value')
                            ], className='trade-stat')
                        ], className='trade-stats')
                    ], className='trade-card')
                
                trade_activity_cards.append(trade_card)
            
            return performance_cards, portfolio_chart, holdings_cards, trade_activity_cards
            
        except Exception as e:
            print(f"Error updating traders comparison: {str(e)}")
            return [], html.Div(f"Error: {str(e)}"), [], []
    
    def create_signals_table(signals):
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Agent"),
                html.Th("Personality"),
                html.Th("Action"),
                html.Th("Commentary"),
                html.Th("Confidence"),
                html.Th("Wallet Value"),
                html.Th("Holdings"),
                html.Th("Return %")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(signal['symbol'].replace('/USDT', '')),
                    html.Td(signal['agent']),
                    html.Td(signal['personality'][:30] + "..."),
                    html.Td(
                        html.Span(
                            signal['signal']['action'].upper(),
                            className=f"signal-{signal['signal']['action'].lower()}"
                        )
                    ),
                    html.Td(
                        html.Div(
                            signal['signal'].get('commentary', ''),
                            className='signal-commentary'
                        )
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
                    html.Td(f"${signal['wallet_metrics']['total_value_usdt']:.2f}"),
                    html.Td(
                        ", ".join([
                            f"{amount:.4f} {sym}"
                            for sym, amount in signal['wallet_metrics']['holdings'].items()
                        ]) or "No holdings"
                    ),
                    html.Td(
                        html.Span(
                            f"{signal['wallet_metrics']['total_return_pct']:+.2f}%",
                            className=f"{'positive' if signal['wallet_metrics']['total_return_pct'] > 0 else 'negative'}"
                        )
                    )
                ], className=f"signal-row-{signal['signal']['action'].lower()}")
                for signal in reversed(signals)
            ])
        ], className='signals-table')
    
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