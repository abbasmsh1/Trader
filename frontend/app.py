"""
Flask application for the trading system frontend.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
from models.database import DatabaseHandler
import json
from datetime import datetime, timedelta
from typing import Dict, Any

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize database handler
db = DatabaseHandler()

# Routes
@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/portfolio')
def portfolio():
    """Render the portfolio page."""
    return render_template('portfolio.html')

@app.route('/history')
def history():
    """Render the history page."""
    return render_template('history.html')

@app.route('/settings')
def settings():
    """Render the settings page."""
    return render_template('settings.html')

@app.route('/api/traders')
def get_traders():
    """Get list of available traders."""
    try:
        traders = db.get_traders()
        return jsonify({'traders': traders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/<trader_id>')
def get_portfolio(trader_id):
    """Get portfolio for a specific trader."""
    try:
        portfolio = db.get_agent_portfolio(trader_id)
        if portfolio:
            return jsonify(portfolio)
        return jsonify({"error": "Portfolio not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/prices')
def get_prices():
    """Get current market prices."""
    try:
        prices = db.get_market_prices()
        return jsonify(prices)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/historical-prices/<symbol>')
def get_historical_prices(symbol):
    """Get historical price data for a symbol."""
    try:
        # Convert URL-encoded symbol (e.g., 'BTC/USDT' or 'BTC-USDT') to proper format
        symbol = symbol.replace('-', '/').upper()
        interval = request.args.get('interval', '1h')
        prices = db.get_historical_prices(symbol, interval)
        return jsonify(prices)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/trade', methods=['POST'])
def execute_trade():
    """Execute a trade."""
    try:
        data = request.json
        result = db.execute_trade(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/history/<agent_id>')
def get_portfolio_history(agent_id):
    """Get portfolio history for an agent."""
    try:
        symbol = request.args.get('symbol')
        limit = request.args.get('limit', default=100, type=int)
        history = db.get_portfolio_history(agent_id, symbol=symbol, limit=limit)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history/<trader_id>')
def get_trade_history(trader_id: str) -> Dict[str, Any]:
    """Get trade history for a trader."""
    try:
        date_range = int(request.args.get('date_range', '30'))
        trades = db.get_trades(trader_id) if trader_id != 'all' else db.get_all_trades()
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=date_range)
        trades = [trade for trade in trades if datetime.fromisoformat(trade['timestamp']) >= cutoff_date]
        
        return {
            'trades': trades,
            'total': len(trades)
        }
    except Exception as e:
        print(f"Error getting trade history: {str(e)}")
        return {'trades': [], 'total': 0}

@app.route('/api/signals/<trader_id>')
def get_signals(trader_id):
    """Get recent trading signals for a trader."""
    try:
        limit = request.args.get('limit', 100, type=int)
        signals = db.get_signals(trader_id, limit)
        return jsonify(signals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals/<trader_id>/active')
def get_active_signals(trader_id):
    """Get active trading signals for a trader."""
    try:
        signals = db.get_active_signals(trader_id)
        return jsonify(signals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/<trader_id>')
def get_performance(trader_id):
    """Get performance metrics for a trader."""
    try:
        limit = request.args.get('limit', default=100, type=int)
        metrics = db.get_performance_history(trader_id, limit=limit)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000) 