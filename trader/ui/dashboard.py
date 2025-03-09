import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from trader.core.trading_system import TradingSystem

def create_dashboard(trading_system: TradingSystem, title: str = "AI Trader Dashboard", subtitle: str = "Cryptocurrency Trading Arena"):
    """
    Create a Dash dashboard for the trading system.
    
    Args:
        trading_system: Trading system instance
        title: Dashboard title
        subtitle: Dashboard subtitle
        
    Returns:
        Dash app instance
    """
    # Get the path to the assets directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_path = os.path.join(os.path.dirname(current_dir), 'assets')
    
    # Create Dash app
    app = dash.Dash(__name__, assets_folder=assets_path)
    
    # Define layout
    app.layout = html.Div(
        className="dashboard-container",
        children=[
            # Header
            html.Header(
                className="header",
                children=[
                    html.Div(
                        className="header-content",
                        children=[
                            html.Div(
                                children=[
                                    html.H1(title, className="dashboard-title"),
                                    html.H2(subtitle, className="dashboard-subtitle")
                                ]
                            ),
                            html.Div(
                                className="header-stats",
                                children=[
                                    html.Div(
                                        id="btc-price",
                                        className="btc-price"
                                    ),
                                    html.Div(
                                        className="live-badge",
                                        children="LIVE"
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Main content
            html.Div(
                className="main-content",
                children=[
                    # Navigation
                    html.Div(
                        className="nav-container",
                        children=[
                            html.Button("Market Overview", id="nav-market", className="nav-button active"),
                            html.Button("Trader Portfolios", id="nav-traders", className="nav-button"),
                            html.Button("Agent Discussions", id="nav-discussions", className="nav-button"),
                            html.Button("Trading Signals", id="nav-signals", className="nav-button")
                        ]
                    ),
                    
                    # Content tabs
                    html.Div(
                        id="content-market",
                        className="content-tab",
                        children=[
                            html.Div(
                                className="section-header",
                                children=[
                                    html.H2("Market Overview", className="section-title"),
                                    html.Div(
                                        className="refresh-indicator",
                                        children=[
                                            html.Span("Last updated: ", className="refresh-label"),
                                            html.Span(id="market-refresh-time", className="refresh-time"),
                                            html.Div(id="market-refresh-spinner", className="refresh-spinner")
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                className="chart-grid",
                                children=[
                                    html.Div(
                                        className="chart-container",
                                        children=[
                                            dcc.Graph(id="price-chart")
                                        ]
                                    ),
                                    html.Div(
                                        className="chart-container",
                                        children=[
                                            dcc.Graph(id="volume-chart")
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                className="control-panel",
                                children=[
                                    html.Div(
                                        className="dropdown",
                                        children=[
                                            html.Label("Symbol"),
                                            dcc.Dropdown(
                                                id="symbol-dropdown",
                                                options=[
                                                    {"label": symbol, "value": symbol}
                                                    for symbol in trading_system.symbols
                                                ],
                                                value=trading_system.symbols[0] if trading_system.symbols else "BTC/USDT"
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="dropdown",
                                        children=[
                                            html.Label("Timeframe"),
                                            dcc.Dropdown(
                                                id="timeframe-dropdown",
                                                options=[
                                                    {"label": "1 Hour", "value": "1h"},
                                                    {"label": "4 Hours", "value": "4h"},
                                                    {"label": "1 Day", "value": "1d"}
                                                ],
                                                value="1h"
                                            )
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                id="market-table-container",
                                className="market-overview-table-container"
                            )
                        ],
                        style={"display": "block"}
                    ),
                    
                    html.Div(
                        id="content-traders",
                        className="content-tab",
                        children=[
                            html.Div(
                                className="section-header",
                                children=[
                                    html.H2("Trader Portfolios", className="section-title"),
                                    html.Div(
                                        className="refresh-indicator",
                                        children=[
                                            html.Span("Last updated: ", className="refresh-label"),
                                            html.Span(id="traders-refresh-time", className="refresh-time"),
                                            html.Div(id="traders-refresh-spinner", className="refresh-spinner")
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                id="performance-cards-container",
                                className="performance-cards-grid"
                            )
                        ],
                        style={"display": "none"}
                    ),
                    
                    html.Div(
                        id="content-discussions",
                        className="content-tab",
                        children=[
                            html.Div(
                                className="section-header",
                                children=[
                                    html.H2("Agent Discussions", className="section-title"),
                                    html.Div(
                                        className="refresh-indicator",
                                        children=[
                                            html.Span("Last updated: ", className="refresh-label"),
                                            html.Span(id="discussions-refresh-time", className="refresh-time"),
                                            html.Div(id="discussions-refresh-spinner", className="refresh-spinner")
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                id="discussions-container",
                                className="discussions-panel"
                            )
                        ],
                        style={"display": "none"}
                    ),
                    
                    html.Div(
                        id="content-signals",
                        className="content-tab",
                        children=[
                            html.Div(
                                className="section-header",
                                children=[
                                    html.H2("Trading Signals", className="section-title"),
                                    html.Div(
                                        className="refresh-indicator",
                                        children=[
                                            html.Span("Last updated: ", className="refresh-label"),
                                            html.Span(id="signals-refresh-time", className="refresh-time"),
                                            html.Div(id="signals-refresh-spinner", className="refresh-spinner")
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                id="signals-container",
                                className="signals-table-container"
                            )
                        ],
                        style={"display": "none"}
                    ),
                    
                    # Auto-refresh interval
                    dcc.Interval(
                        id="auto-refresh",
                        interval=30 * 1000,  # 30 seconds
                        n_intervals=0
                    ),
                    
                    # Trade history modal
                    html.Div(
                        id="trade-history-modal",
                        className="modal-backdrop",
                        children=[
                            html.Div(
                                className="modal-container",
                                children=[
                                    html.Div(
                                        className="modal-header",
                                        children=[
                                            html.H3(id="modal-title", className="modal-title"),
                                            html.Button("×", id="modal-close", className="modal-close")
                                        ]
                                    ),
                                    html.Div(
                                        id="modal-content",
                                        className="modal-content"
                                    )
                                ]
                            )
                        ],
                        style={"display": "none"}
                    ),
                    
                    # Memory status indicator
                    html.Div(
                        id="memory-status",
                        className="memory-status",
                        children=[
                            html.Div(className="pulse-ring"),
                            html.Span(id="memory-status-text")
                        ]
                    )
                ]
            )
        ]
    )
    
    # Define callbacks
    
    # Tab navigation
    @app.callback(
        [
            Output("content-market", "style"),
            Output("content-traders", "style"),
            Output("content-discussions", "style"),
            Output("content-signals", "style"),
            Output("nav-market", "className"),
            Output("nav-traders", "className"),
            Output("nav-discussions", "className"),
            Output("nav-signals", "className")
        ],
        [
            Input("nav-market", "n_clicks"),
            Input("nav-traders", "n_clicks"),
            Input("nav-discussions", "n_clicks"),
            Input("nav-signals", "n_clicks")
        ]
    )
    def toggle_tabs(market_clicks, traders_clicks, discussions_clicks, signals_clicks):
        ctx = callback_context
        
        if not ctx.triggered:
            # Default to market tab
            return (
                {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"},
                "nav-button active", "nav-button", "nav-button", "nav-button"
            )
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "nav-market":
            return (
                {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"},
                "nav-button active", "nav-button", "nav-button", "nav-button"
            )
        elif button_id == "nav-traders":
            return (
                {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"},
                "nav-button", "nav-button active", "nav-button", "nav-button"
            )
        elif button_id == "nav-discussions":
            return (
                {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"},
                "nav-button", "nav-button", "nav-button active", "nav-button"
            )
        elif button_id == "nav-signals":
            return (
                {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"},
                "nav-button", "nav-button", "nav-button", "nav-button active"
            )
            
    # Update BTC price
    @app.callback(
        Output("btc-price", "children"),
        Input("auto-refresh", "n_intervals")
    )
    def update_btc_price(n):
        price = trading_system.data_fetcher.fetch_current_price("BTC/USDT")
        return f"BTC: ${price:.2f}"
        
    # Update price chart
    @app.callback(
        Output("price-chart", "figure"),
        [
            Input("auto-refresh", "n_intervals"),
            Input("symbol-dropdown", "value"),
            Input("timeframe-dropdown", "value")
        ]
    )
    def update_price_chart(n, symbol, timeframe):
        # Get market data
        data = trading_system.get_market_data(symbol, timeframe)
        
        if data.empty:
            return go.Figure()
            
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price"
            )
        )
        
        # Add Bollinger Bands if available
        if "bb_upper" in data.columns and "bb_middle" in data.columns and "bb_lower" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["bb_upper"],
                    name="Upper BB",
                    line=dict(color="rgba(255, 255, 255, 0.5)"),
                    hoverinfo="skip"
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["bb_middle"],
                    name="Middle BB",
                    line=dict(color="rgba(255, 255, 255, 0.5)"),
                    hoverinfo="skip"
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["bb_lower"],
                    name="Lower BB",
                    line=dict(color="rgba(255, 255, 255, 0.5)"),
                    fill="tonexty",
                    fillcolor="rgba(255, 255, 255, 0.1)",
                    hoverinfo="skip"
                )
            )
            
        # Update layout
        fig.update_layout(
            title=f"{symbol} Price Chart ({timeframe})",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    # Update volume chart
    @app.callback(
        Output("volume-chart", "figure"),
        [
            Input("auto-refresh", "n_intervals"),
            Input("symbol-dropdown", "value"),
            Input("timeframe-dropdown", "value")
        ]
    )
    def update_volume_chart(n, symbol, timeframe):
        # Get market data
        data = trading_system.get_market_data(symbol, timeframe)
        
        if data.empty:
            return go.Figure()
            
        # Create figure
        fig = go.Figure()
        
        # Add volume bars
        colors = ["red" if data["close"].iloc[i] < data["open"].iloc[i] else "green" for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                name="Volume",
                marker_color=colors
            )
        )
        
        # Add RSI if available
        if "rsi" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["rsi"],
                    name="RSI",
                    yaxis="y2",
                    line=dict(color="yellow")
                )
            )
            
            # Add RSI reference lines
            fig.add_shape(
                type="line",
                x0=data.index[0],
                y0=70,
                x1=data.index[-1],
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                yref="y2"
            )
            
            fig.add_shape(
                type="line",
                x0=data.index[0],
                y0=30,
                x1=data.index[-1],
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                yref="y2"
            )
            
        # Update layout
        fig.update_layout(
            title=f"{symbol} Volume and Indicators ({timeframe})",
            xaxis_title="Time",
            yaxis_title="Volume",
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100]
            )
        )
        
        return fig
        
    # Update market table
    @app.callback(
        Output("market-table-container", "children"),
        Input("auto-refresh", "n_intervals")
    )
    def update_market_table(n):
        # Get current prices for all symbols
        prices = {}
        
        for symbol in trading_system.symbols:
            price = trading_system.data_fetcher.fetch_current_price(symbol)
            
            # Check if price is valid (not None and greater than 0)
            if price is not None and price > 0:
                # Get 24h data
                data = trading_system.get_market_data(symbol, "1d", limit=2)
                
                if not data.empty and len(data) >= 2:
                    prev_price = data["close"].iloc[-2]
                    change_pct = (price - prev_price) / prev_price * 100
                else:
                    change_pct = 0
                    
                prices[symbol] = {
                    "price": price,
                    "change_pct": change_pct
                }
                
        # Create table
        table = html.Table(
            className="market-overview-table",
            children=[
                html.Thead(
                    html.Tr([
                        html.Th("Symbol"),
                        html.Th("Price"),
                        html.Th("24h Change")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(symbol),
                        html.Td(f"${prices[symbol]['price']:.2f}"),
                        html.Td(
                            f"{prices[symbol]['change_pct']:.2f}%",
                            className="positive" if prices[symbol]['change_pct'] >= 0 else "negative"
                        )
                    ])
                    for symbol in sorted(prices.keys())
                ])
            ]
        )
        
        return table
        
    # Update trader performance cards
    @app.callback(
        Output("performance-cards-container", "children"),
        Input("auto-refresh", "n_intervals")
    )
    def update_traders_comparison(n):
        # Get agent performance metrics
        metrics_list = trading_system.get_agent_performance()
        
        # Create cards
        cards = []
        
        for metrics in metrics_list:
            # Get basic metrics
            name = metrics.get("name", "Unknown")
            personality = metrics.get("personality", "")
            total_value = metrics.get("total_value", 0)
            balance = metrics.get("balance_usdt", 0)
            profit_loss = metrics.get("profit_loss", 0)
            profit_loss_pct = metrics.get("profit_loss_pct", 0)
            
            # Get holdings
            holdings = metrics.get("holdings_with_prices", {})
            
            # Create holdings list
            holdings_list = []
            
            for symbol, holding in holdings.items():
                # Check if stop loss exists
                stop_loss_info = ""
                if "stop_loss" in holding:
                    stop_loss_price = holding["stop_loss"]
                    stop_loss_pct = holding.get("stop_loss_pct", 5.0)
                    stop_loss_info = html.Span(
                        f" (SL: ${stop_loss_price:.2f}, {stop_loss_pct:.1f}%)",
                        className="stop-loss-info"
                    )
                
                holdings_list.append(
                    html.Div(
                        className="holding-row",
                        children=[
                            html.Div(
                                className="holding-symbol",
                                children=symbol
                            ),
                            html.Div(
                                className="holding-amount",
                                children=f"{holding['amount']:.8f}"
                            ),
                            html.Div(
                                className="holding-value",
                                children=[
                                    f"${holding['value_usdt']:.2f}",
                                    stop_loss_info
                                ]
                            )
                        ]
                    )
                )
                
            # Create card
            card = html.Div(
                className="performance-card",
                children=[
                    html.Div(
                        className="card-content",
                        children=[
                            html.Div(
                                className="agent-name",
                                children=[
                                    name,
                                    html.Span(
                                        personality.split(" - ")[0] if " - " in personality else personality,
                                        className="agent-badge"
                                    )
                                ]
                            ),
                            html.Div(
                                className="total-value",
                                children=[
                                    f"${total_value:.2f}",
                                    html.Span(
                                        f"({profit_loss_pct:+.2f}%)",
                                        className="change positive" if profit_loss_pct >= 0 else "change negative"
                                    )
                                ]
                            ),
                            html.Div(
                                className="balance-container",
                                children=[
                                    html.Div(
                                        className="balance-row",
                                        children=[
                                            html.Div("USDT Balance", className="balance-label"),
                                            html.Div(f"${balance:.2f}", className="balance-value")
                                        ]
                                    )
                                ]
                            ),
                            html.Div(
                                className="holdings-container",
                                children=[
                                    html.Div("Holdings", className="holdings-title"),
                                    html.Div(
                                        className="holdings-list",
                                        children=holdings_list if holdings_list else "No holdings"
                                    )
                                ]
                            ),
                            html.Div(
                                className="trades-container",
                                children=[
                                    html.Div(
                                        className="trades-header",
                                        children=[
                                            html.Div("Recent Trades", className="trades-title"),
                                            html.Button(
                                                "View All",
                                                id={"type": "view-trades", "agent": name},
                                                className="view-all-btn"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
            
            cards.append(card)
            
        return cards
        
    # Update discussions panel
    @app.callback(
        Output("discussions-container", "children"),
        Input("auto-refresh", "n_intervals")
    )
    def update_discussions(n):
        # Get discussions history
        discussions = trading_system.discussions_history
        
        if not discussions:
            return html.Div("No discussions yet")
            
        # Sort by timestamp (newest first)
        discussions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Create discussion blocks
        blocks = []
        
        for discussion in discussions:
            symbol = discussion.get("symbol", "Unknown")
            timestamp = discussion.get("timestamp")
            price = discussion.get("price", 0)
            messages = discussion.get("messages", [])
            
            # Create message elements
            message_elements = []
            
            for message in messages:
                agent = message.get("agent", "Unknown")
                text = message.get("message", "")
                
                message_elements.append(
                    html.Div(
                        className="discussion-message",
                        children=[
                            html.Strong(f"{agent}: "),
                            text
                        ]
                    )
                )
                
            # Create discussion block
            block = html.Div(
                className="discussion-block",
                children=[
                    html.Div(
                        className="discussion-header",
                        children=[
                            html.Div(symbol, className="discussion-symbol"),
                            html.Div(
                                format_timestamp(timestamp),
                                className="discussion-time"
                            )
                        ]
                    ),
                    html.Div(
                        className="discussion-messages",
                        children=message_elements
                    )
                ]
            )
            
            blocks.append(block)
            
        return blocks
        
    # Update signals table
    @app.callback(
        Output("signals-container", "children"),
        Input("auto-refresh", "n_intervals")
    )
    def update_signals(n):
        # Get signals history
        signals = trading_system.signals_history
        
        if not signals:
            return html.Div("No signals yet", className="no-data-message")
            
        # Sort by timestamp (newest first)
        signals = sorted(signals, key=lambda x: x.get("timestamp", 0), reverse=True)
        
        # Create table
        table = html.Table(
            className="signals-table",
            children=[
                html.Thead(
                    html.Tr([
                        html.Th("Time"),
                        html.Th("Agent"),
                        html.Th("Symbol"),
                        html.Th("Action"),
                        html.Th("Confidence"),
                        html.Th("Reason")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(format_timestamp(signal.get("timestamp"))),
                        html.Td(signal.get("agent_name", "")),
                        html.Td(signal.get("symbol", "")),
                        html.Td(
                            signal.get("action", ""),
                            className=f"signal-action {signal.get('action', '').lower()}"
                        ),
                        html.Td(f"{signal.get('confidence', 0):.2f}"),
                        html.Td(signal.get("reason", ""))
                    ]) for signal in signals[:50]  # Show last 50 signals
                ])
            ]
        )
        
        return html.Div([
            html.Div("Signals History", className="section-title"),
            table
        ])
        
    def format_timestamp(timestamp):
        """Format timestamp for display."""
        if isinstance(timestamp, datetime):
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        else:
            return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    # Show trade history modal
    @app.callback(
        [
            Output("trade-history-modal", "style"),
            Output("modal-title", "children"),
            Output("modal-content", "children")
        ],
        [
            Input({"type": "view-trades", "agent": ALL}, "n_clicks"),
            Input("modal-close", "n_clicks")
        ],
        [State("trade-history-modal", "style")]
    )
    def show_trade_history(view_clicks, close_clicks, current_style):
        ctx = callback_context
        
        if not ctx.triggered:
            return current_style or {"display": "none"}, "", ""
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "modal-close":
            return {"display": "none"}, "", ""
            
        try:
            # Parse the button ID to get the agent name
            button_data = eval(button_id)
            agent_name = button_data.get("agent", "")
            
            if not agent_name:
                return current_style or {"display": "none"}, "", ""
                
            # Find the agent
            agent = next((a for a in trading_system.agents if a.name == agent_name), None)
            
            if not agent:
                return current_style or {"display": "none"}, "", ""
                
            # Get trade history
            trade_history_df = agent.wallet.get_trade_history_df()
            
            if trade_history_df.empty:
                return {"display": "block"}, f"{agent_name} Trade History", "No trades yet"
                
            # Sort by timestamp (newest first)
            trade_history_df = trade_history_df.sort_values("timestamp", ascending=False)
            
            # Create table
            table = html.Table(
                className="trade-history-table",
                children=[
                    html.Thead(
                        html.Tr([
                            html.Th("Time"),
                            html.Th("Symbol"),
                            html.Th("Action"),
                            html.Th("Amount"),
                            html.Th("Price"),
                            html.Th("Value")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(format_timestamp(row["timestamp"])),
                            html.Td(row["symbol"]),
                            html.Td(
                                row["action"],
                                className=f"trade-action {row['action'].lower()}"
                            ),
                            html.Td(f"{row['amount_crypto']:.8f}"),
                            html.Td(f"${row['price']:.2f}"),
                            html.Td(f"${row['amount_usdt']:.2f}")
                        ])
                        for _, row in trade_history_df.iterrows()
                    ])
                ]
            )
            
            return {"display": "block"}, f"{agent_name} Trade History", table
            
        except Exception as e:
            print(f"Error showing trade history: {e}")
            return current_style or {"display": "none"}, "", ""
            
    # Update refresh indicators
    @app.callback(
        [
            Output("market-refresh-time", "children"),
            Output("market-refresh-spinner", "className"),
            Output("traders-refresh-time", "children"),
            Output("traders-refresh-spinner", "className"),
            Output("discussions-refresh-time", "children"),
            Output("discussions-refresh-spinner", "className"),
            Output("signals-refresh-time", "children"),
            Output("signals-refresh-spinner", "className"),
            Output("memory-status-text", "children")
        ],
        Input("auto-refresh", "n_intervals")
    )
    def update_refresh_indicators(n):
        now = datetime.now().strftime("%H:%M:%S")
        spinner_class = "refresh-spinner active" if n % 2 == 0 else "refresh-spinner"
        
        # Calculate time since last save
        time_since_save = (datetime.now() - trading_system.last_save).total_seconds()
        save_status = f"Last saved: {trading_system.last_save.strftime('%H:%M:%S')}"
        
        return now, spinner_class, now, spinner_class, now, spinner_class, now, spinner_class, save_status
        
    return app 