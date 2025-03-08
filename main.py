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
import ta

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
    Create a modern and interactive Dash dashboard with multiple coin pairs.
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
        ], className='main-content'),
        
        # Refresh interval
        dcc.Interval(
            id='interval-component',
            interval=SYSTEM_PARAMS['dashboard_refresh_rate'],
            n_intervals=0
        )
    ], className='container')
    
    def create_chart(symbol: str, df: pd.DataFrame, show_indicators: List[str] = None) -> go.Figure:
        """Create a chart for a single symbol with technical indicators."""
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            yaxis='y'
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            yaxis='y3',
            opacity=0.3
        ))
        
        # Calculate and add technical indicators
        if show_indicators:
            # Calculate SMA
            if 'SMA' in show_indicators:
                df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
                df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['SMA200'] = ta.trend.sma_indicator(df['close'], window=200)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='#ff7f0e')))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200', line=dict(color='#2ca02c')))
            
            # Calculate RSI
            if 'RSI' in show_indicators:
                df['RSI'] = ta.momentum.rsi(df['close'], window=14)
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['RSI'],
                    name='RSI',
                    yaxis='y2',
                    line=dict(color='#d62728')
                ))
            
            # Calculate MACD
            if 'MACD' in show_indicators:
                macd = ta.trend.MACD(df['close'])
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_hist'] = macd.macd_diff()
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['MACD'],
                    name='MACD',
                    yaxis='y4',
                    line=dict(color='#9467bd')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['MACD_signal'],
                    name='MACD Signal',
                    yaxis='y4',
                    line=dict(color='#e377c2')
                ))
                fig.add_trace(go.Bar(
                    x=df.index, y=df['MACD_hist'],
                    name='MACD Histogram',
                    yaxis='y4',
                    marker_color='#17becf'
                ))
            
            # Calculate Bollinger Bands
            if 'BB' in show_indicators:
                bollinger = ta.volatility.BollingerBands(df['close'])
                df['BB_upper'] = bollinger.bollinger_hband()
                df['BB_lower'] = bollinger.bollinger_lband()
                df['BB_mid'] = bollinger.bollinger_mavg()
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['BB_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['BB_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash'),
                    fill='tonexty'
                ))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title=f'{symbol.replace("/USDT", "")} Price & Indicators',
            yaxis=dict(
                title='Price (USDT)',
                domain=[0.4, 1]
            ),
            yaxis2=dict(
                title='RSI',
                domain=[0.2, 0.38],
                range=[0, 100]
            ),
            yaxis3=dict(
                title='Volume',
                domain=[0, 0.18]
            ),
            yaxis4=dict(
                title='MACD',
                domain=[0.2, 0.38]
            ),
            height=600,
            template='plotly_dark',
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(43, 60, 78, 0.8)'
            )
        )
        
        # Add price statistics
        latest = df.iloc[-1]
        prev_day = df.iloc[-24] if timeframe == '1h' else df.iloc[-1]
        price_change = (latest['close'] - prev_day['close']) / prev_day['close'] * 100
        
        stats_text = f"""
        Current Price: ${latest['close']:,.2f}
        24h Change: {price_change:+.2f}%
        24h High: ${df['high'][-24:].max():,.2f}
        24h Low: ${df['low'][-24:].min():,.2f}
        24h Volume: ${df['volume'][-24:].sum():,.0f}
        """
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            showarrow=False,
            font=dict(size=12),
            bgcolor='rgba(43, 60, 78, 0.8)',
            bordercolor='rgba(68, 86, 105, 0.8)',
            borderwidth=1,
            align='left'
        )
        
        return fig
    
    @app.callback(
        [Output('multi-chart-container', 'children'),
         Output('signals-table', 'children'),
         Output('performance-metrics', 'children'),
         Output('market-overview', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('symbol-multi-dropdown', 'value'),
         Input('timeframe-dropdown', 'value'),
         Input('layout-radio', 'value'),
         Input('indicator-checklist', 'value'),
         Input('chart-style', 'value'),
         Input('refresh-interval-slider', 'value')],
        [State('auto-refresh-switch', 'value')]
    )
    def update_dashboard(n, symbols, timeframe, layout, indicators, chart_style, refresh_interval, auto_refresh):
        if not auto_refresh or (n > 0 and 'enabled' not in auto_refresh):
            raise dash.exceptions.PreventUpdate
        
        if not symbols:
            return (
                html.Div("Please select at least one cryptocurrency", className='no-data-message'),
                None,
                None,
                None
            )
            
        # Create charts for each symbol
        charts = []
        all_signals = []
        performance_data = []
        market_data = []
        
        for symbol in symbols:
            # Get market data
            df = trading_system.data_fetcher.get_historical_data(symbol, timeframe)
            
            # Create chart
            fig = create_chart(symbol, df, indicators)
            
            # Create chart container
            chart_div = html.Div(
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False}
                ),
                className=f'chart-container-{layout}'
            )
            charts.append(chart_div)
            
            # Get signals for this symbol
            recent_signals = [s for s in trading_system.signals_history 
                            if s['symbol'] == symbol][-3:]
            all_signals.extend(recent_signals)
            
            # Calculate performance metrics
            latest = df.iloc[-1]
            prev_day = df.iloc[-24] if timeframe == '1h' else df.iloc[-1]
            price_change = (latest['close'] - prev_day['close']) / prev_day['close'] * 100
            
            performance_data.append({
                'symbol': symbol.replace('/USDT', ''),
                'price': latest['close'],
                'change_24h': price_change,
                'volume_24h': df['volume'][-24:].sum(),
                'high_24h': df['high'][-24:].max(),
                'low_24h': df['low'][-24:].min()
            })
            
            # Calculate market metrics
            volatility = df['close'].pct_change().std() * 100
            avg_volume = df['volume'].mean()
            market_data.append({
                'symbol': symbol.replace('/USDT', ''),
                'volatility': volatility,
                'avg_volume': avg_volume,
                'trend': 'Bullish' if price_change > 0 else 'Bearish'
            })
        
        # Create the chart grid
        chart_grid = html.Div(charts, className=f'chart-grid-{layout}')
        
        # Create signals table
        signals_table = create_signals_table(all_signals)
        
        # Create performance metrics
        performance_metrics = create_performance_metrics(performance_data)
        
        # Create market overview
        market_overview = create_market_overview(market_data)
        
        return chart_grid, signals_table, performance_metrics, market_overview
    
    def create_signals_table(signals):
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Agent"),
                html.Th("Action"),
                html.Th("Confidence"),
                html.Th("Price")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(signal['symbol'].replace('/USDT', '')),
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