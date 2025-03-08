import os
from typing import List, Dict
import pandas as pd
from datetime import datetime
import time
from data.market_data import MarketDataFetcher
from agents.trend_follower import TrendFollower
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from threading import Thread
import plotly.io as pio
from config.settings import DEFAULT_SYMBOLS, SYSTEM_PARAMS, TIMEFRAMES

# Set default plotly theme
pio.templates.default = "plotly_dark"

class TradingSystem:
    def __init__(self, symbols: List[str] = None):
        """
        Initialize the trading system.
        
        Args:
            symbols (List[str]): List of trading pairs to monitor
        """
        self.data_fetcher = MarketDataFetcher()
        self.symbols = symbols or DEFAULT_SYMBOLS
        
        # Initialize trading agents
        self.agents = [
            TrendFollower(name="Conservative Trend Follower", risk_tolerance=0.3),
            TrendFollower(name="Moderate Trend Follower", risk_tolerance=0.6),
            TrendFollower(name="Aggressive Trend Follower", risk_tolerance=0.8)
        ]
        
        self.signals_history = []
        
    def analyze_market(self, symbol: str) -> List[Dict]:
        """
        Analyze market data using all agents.
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            List[Dict]: List of signals from all agents
        """
        # Fetch market data
        market_data = self.data_fetcher.get_historical_data(symbol)
        if market_data.empty:
            return []
        
        signals = []
        for agent in self.agents:
            # Get analysis and signal from each agent
            analysis = agent.analyze_market(market_data)
            signal = agent.generate_signal(analysis)
            
            signals.append({
                'agent': agent.name,
                'symbol': symbol,
                'signal': signal,
                'risk_tolerance': agent.risk_tolerance
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
    Create a modern and interactive Dash dashboard.
    
    Args:
        trading_system (TradingSystem): The trading system instance
    """
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
    )
    
    # Define colors
    colors = {
        'background': '#1f2630',
        'text': '#ffffff',
        'grid': '#283442',
        'panel': '#2b3c4e'
    }
    
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("AI Crypto Trading Dashboard", className='dashboard-title'),
            html.P("Real-time cryptocurrency trading signals powered by AI", className='dashboard-subtitle')
        ], className='header'),
        
        # Main content
        html.Div([
            # Left panel - Controls
            html.Div([
                html.Div([
                    html.H3("Trading Controls", className='panel-title'),
                    html.Label("Select Cryptocurrency"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': s.replace('/USDT', ''), 'value': s} 
                                for s in trading_system.symbols],
                        value=trading_system.symbols[0],
                        className='dropdown'
                    ),
                    html.Label("Timeframe"),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[{'label': v, 'value': k} for k, v in TIMEFRAMES.items()],
                        value='1h',
                        className='dropdown'
                    ),
                    html.Div([
                        html.Label("Auto Refresh"),
                        dcc.Checklist(
                            id='auto-refresh-switch',
                            options=[{'label': '', 'value': 'enabled'}],
                            value=['enabled'],
                            className='auto-refresh-toggle'
                        )
                    ], className='switch-container')
                ], className='control-panel')
            ], className='left-panel'),
            
            # Right panel - Charts and Data
            html.Div([
                # Price Chart
                html.Div([
                    html.H3("Price Chart", className='panel-title'),
                    dcc.Graph(
                        id='price-chart',
                        config={'displayModeBar': True}
                    )
                ], className='chart-panel'),
                
                # Trading Signals
                html.Div([
                    html.H3("Latest Trading Signals", className='panel-title'),
                    html.Div(id='signals-table', className='signals-table')
                ], className='signals-panel')
            ], className='right-panel')
        ], className='main-content'),
        
        # Refresh interval
        dcc.Interval(
            id='interval-component',
            interval=SYSTEM_PARAMS['dashboard_refresh_rate'],
            n_intervals=0
        )
    ], className='container')
    
    @app.callback(
        [Output('price-chart', 'figure'),
         Output('signals-table', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('symbol-dropdown', 'value'),
         Input('timeframe-dropdown', 'value')],
        [State('auto-refresh-switch', 'value')]
    )
    def update_charts(n, symbol, timeframe, auto_refresh):
        if not auto_refresh or (n > 0 and 'enabled' not in auto_refresh):  # Don't update if auto-refresh is off
            raise dash.exceptions.PreventUpdate
            
        # Get market data
        df = trading_system.data_fetcher.get_historical_data(symbol, timeframe)
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Chart',
            yaxis_title='Price (USDT)',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            height=600,
            template='plotly_dark',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            xaxis_rangeslider_visible=False
        )
        
        # Get recent signals
        recent_signals = [s for s in trading_system.signals_history 
                         if s['symbol'] == symbol][-10:]
        
        # Create signals table with modern styling
        signals_table = html.Table([
            html.Thead(html.Tr([
                html.Th("Agent"),
                html.Th("Action"),
                html.Th("Confidence"),
                html.Th("Price")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(signal['agent']),
                    html.Td(
                        html.Span(
                            signal['signal']['action'].upper(),
                            className=f"signal-{signal['signal']['action'].lower()}"
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
                    html.Td(f"${signal['signal']['price']:,.2f}")
                ], className=f"signal-row-{signal['signal']['action'].lower()}")
                for signal in reversed(recent_signals)
            ])
        ], className='signals-table')
        
        return fig, signals_table
    
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
                /* Responsive design */
                @media (max-width: 1024px) {
                    .main-content {
                        flex-direction: column;
                    }
                    .left-panel {
                        max-width: none;
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