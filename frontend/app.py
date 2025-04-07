"""
Flask application serving as the frontend for the Crypto Trader system.

This provides a web interface to monitor the system's performance,
view portfolio status, and control the trading system.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from threading import Thread
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO

# Import trading system components
from models.wallet import Wallet
try:
    from agents.system_controller import SystemControllerAgent
    trading_system_available = True
except ImportError:
    trading_system_available = False
    print("Warning: Trading system components not available, running in demo mode")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("frontend")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'crypto_trader_secret'
socketio = SocketIO(app)

# Global variables
wallet = None
system_controller = None
system_running = False
system_thread = None
demo_mode = not trading_system_available
last_portfolio_update = datetime.now()
last_prices = {
    "BTC": 50000.0,
    "ETH": 3000.0,
    "SOL": 100.0,
    "BNB": 400.0
}

# Initialize demo wallet if needed
if demo_mode:
    wallet = Wallet(initial_balance=10000.0, base_currency="USDT", name="Demo Wallet")
    # Add some demo holdings
    wallet.add_deposit("BTC", 0.1, source="external")
    wallet.add_deposit("ETH", 1.0, source="external")
    wallet.update_prices(last_prices)


def load_system_controller():
    """Load the system controller if the trading system is available."""
    global system_controller, wallet
    
    if not trading_system_available:
        return False
    
    try:
        # Load configuration
        from config.config import load_config
        config = load_config()
        
        # Initialize system controller
        system_controller = SystemControllerAgent(
            name="System Controller",
            description="Main controller for the trading system",
            config=config,
            parent_id=None
        )
        
        # Get wallet from system controller if available
        if hasattr(system_controller, 'wallet'):
            wallet = system_controller.wallet
        else:
            # Create a wallet if system doesn't have one
            wallet = Wallet(
                initial_balance=config.get("initial_balance", 10000.0),
                base_currency=config.get("base_currency", "USDT"),
                name="Trading Wallet"
            )
            if hasattr(system_controller, 'set_wallet'):
                system_controller.set_wallet(wallet)
                
        return True
    except Exception as e:
        logger.error(f"Error loading system controller: {str(e)}")
        return False


def update_demo_prices():
    """Update demo prices with small random variations."""
    global last_prices
    import random
    
    for symbol in last_prices:
        # Simulate small price movements (±1%)
        change = random.uniform(-0.01, 0.01)
        last_prices[symbol] *= (1 + change)
    
    # Update wallet with new prices
    if wallet:
        wallet.update_prices(last_prices)


def background_task():
    """Background task to run the trading system or update demo data."""
    global system_running, last_portfolio_update
    
    logger.info("Starting background task")
    
    while system_running:
        try:
            # For demo mode, just update prices periodically
            if demo_mode:
                update_demo_prices()
                # Calculate portfolio value
                if wallet:
                    valuation = wallet.calculate_total_value(last_prices)
                    portfolio_data = {
                        "timestamp": datetime.now().isoformat(),
                        "total_value": valuation["total_value"],
                        "holdings": valuation["holdings"]
                    }
                    socketio.emit('portfolio_update', portfolio_data)
            
            # For real trading system, run a cycle
            elif system_controller:
                # Run a single cycle
                result = system_controller.run({})
                if result and "portfolio" in result:
                    socketio.emit('portfolio_update', result["portfolio"])
                    socketio.emit('system_status', result["status"])
            
            # Emit price updates
            socketio.emit('price_update', last_prices)
            
            # Only update every 5 seconds to avoid flooding
            last_portfolio_update = datetime.now()
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in background task: {str(e)}")
            time.sleep(10)  # Wait longer after an error
    
    logger.info("Background task stopped")


@app.route('/')
def index():
    """Render the dashboard page."""
    return render_template('index.html', demo_mode=demo_mode)


@app.route('/portfolio')
def portfolio():
    """Render the portfolio page."""
    if not wallet:
        return redirect(url_for('index'))
    
    balances = wallet.get_all_balances()
    valuation = wallet.calculate_total_value(last_prices)
    
    return render_template(
        'portfolio.html',
        balances=balances,
        valuation=valuation,
        demo_mode=demo_mode
    )


@app.route('/trades')
def trades():
    """Render the trade history page."""
    if not wallet:
        return redirect(url_for('index'))
    
    trade_history = wallet.get_trade_history(limit=50)
    
    return render_template(
        'trades.html',
        trades=trade_history,
        demo_mode=demo_mode
    )


@app.route('/settings')
def settings():
    """Render the settings page."""
    return render_template('settings.html', demo_mode=demo_mode)


@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint to get portfolio data."""
    if not wallet:
        return jsonify({"error": "Wallet not initialized"})
    
    balances = wallet.get_all_balances()
    valuation = wallet.calculate_total_value(last_prices)
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "balances": balances,
        "valuation": valuation,
        "total_value": valuation["total_value"]
    })


@app.route('/api/trades')
def api_trades():
    """API endpoint to get trade history."""
    if not wallet:
        return jsonify({"error": "Wallet not initialized"})
    
    limit = request.args.get('limit', 50, type=int)
    trade_history = wallet.get_trade_history(limit=limit)
    
    return jsonify({
        "trades": trade_history,
        "count": len(trade_history)
    })


@app.route('/api/prices')
def api_prices():
    """API endpoint to get current prices."""
    return jsonify(last_prices)


@app.route('/api/system/start', methods=['POST'])
def api_start_system():
    """API endpoint to start the trading system."""
    global system_running, system_thread
    
    if system_running:
        return jsonify({"status": "already_running"})
    
    system_running = True
    system_thread = Thread(target=background_task)
    system_thread.daemon = True
    system_thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/system/stop', methods=['POST'])
def api_stop_system():
    """API endpoint to stop the trading system."""
    global system_running
    
    if not system_running:
        return jsonify({"status": "not_running"})
    
    system_running = False
    
    # Wait for thread to terminate
    if system_thread and system_thread.is_alive():
        system_thread.join(timeout=5.0)
    
    return jsonify({"status": "stopped"})


@app.route('/api/system/status')
def api_system_status():
    """API endpoint to get system status."""
    if demo_mode:
        return jsonify({
            "status": "running" if system_running else "stopped",
            "mode": "demo",
            "uptime": "N/A",
            "last_update": last_portfolio_update.isoformat()
        })
    
    if not system_controller:
        return jsonify({"error": "System controller not initialized"})
    
    status = system_controller.get_system_status()
    
    return jsonify(status)


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial data
    if wallet:
        valuation = wallet.calculate_total_value(last_prices)
        socketio.emit('portfolio_update', {
            "timestamp": datetime.now().isoformat(),
            "total_value": valuation["total_value"],
            "holdings": valuation["holdings"]
        }, room=request.sid)
    
    socketio.emit('price_update', last_prices, room=request.sid)
    
    # Send system status
    if demo_mode:
        socketio.emit('system_status', {
            "status": "running" if system_running else "stopped",
            "mode": "demo"
        }, room=request.sid)
    elif system_controller:
        socketio.emit('system_status', system_controller.get_system_status(), room=request.sid)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


# Demo trade endpoints
@app.route('/api/demo/buy', methods=['POST'])
def api_demo_buy():
    """API endpoint to execute a demo buy trade."""
    if not demo_mode or not wallet:
        return jsonify({"error": "Not in demo mode or wallet not initialized"})
    
    data = request.json
    symbol = data.get('symbol')
    amount_usd = float(data.get('amount', 100.0))
    
    if not symbol or symbol not in last_prices:
        return jsonify({"error": f"Invalid symbol: {symbol}"})
    
    price = last_prices[symbol]
    quantity = amount_usd / price
    
    try:
        trade = wallet.add_trade(
            trade_type="buy",
            from_currency="USDT",
            to_currency=symbol,
            from_amount=amount_usd,
            to_amount=quantity,
            price=price,
            fee=amount_usd * 0.001,
            exchange="demo"
        )
        
        return jsonify({
            "success": True,
            "trade_id": trade["id"],
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "amount": amount_usd
        })
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/api/demo/sell', methods=['POST'])
def api_demo_sell():
    """API endpoint to execute a demo sell trade."""
    if not demo_mode or not wallet:
        return jsonify({"error": "Not in demo mode or wallet not initialized"})
    
    data = request.json
    symbol = data.get('symbol')
    quantity = float(data.get('quantity', 0))
    
    if not symbol or symbol not in last_prices:
        return jsonify({"error": f"Invalid symbol: {symbol}"})
    
    # If quantity is not specified, sell all
    if quantity <= 0:
        quantity = wallet.get_balance(symbol)
    
    if quantity <= 0:
        return jsonify({"error": f"No {symbol} balance to sell"})
    
    price = last_prices[symbol]
    amount_usd = quantity * price
    
    try:
        trade = wallet.add_trade(
            trade_type="sell",
            from_currency=symbol,
            to_currency="USDT",
            from_amount=quantity,
            to_amount=amount_usd,
            price=price,
            fee=amount_usd * 0.001,
            exchange="demo"
        )
        
        return jsonify({
            "success": True,
            "trade_id": trade["id"],
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "amount": amount_usd
        })
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == '__main__':
    # Initialize system
    if not demo_mode:
        load_system_controller()
    
    # Run the server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 