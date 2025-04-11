"""
Flask application for the trading system frontend.
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
from models.database import DatabaseHandler
import json
import os
import random
import requests
from datetime import datetime, timedelta

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize database handler
db = DatabaseHandler()

# Mock data for development
MOCK_DATA = {
    'portfolio': {
        'total_value': 15000.00,
        'base_currency': 'USDT',
        'base_amount': 5000.00,
        'assets_value': 10000.00,
        'num_assets': 3,
        'holdings': [
            {'symbol': 'BTC', 'amount': 0.15, 'value': 6000.00, 'price': 40000.00, 'change_24h': 2.5},
            {'symbol': 'ETH', 'amount': 2.5, 'value': 3000.00, 'price': 1200.00, 'change_24h': -1.2},
            {'symbol': 'SOL', 'amount': 20, 'value': 1000.00, 'price': 50.00, 'change_24h': 5.7}
        ]
    },
    'market_data': {
        'BTC': {'price': 40000.00, 'change_24h': 2.5},
        'ETH': {'price': 1200.00, 'change_24h': -1.2},
        'SOL': {'price': 50.00, 'change_24h': 5.7},
        'ADA': {'price': 0.50, 'change_24h': 0.8},
        'DOT': {'price': 15.00, 'change_24h': -0.5}
    },
    'traders': [
        {'id': 'buffett', 'name': 'Buffett Trader', 'style': 'Value Investing'},
        {'id': 'soros', 'name': 'Soros Trader', 'style': 'Macro Trading'},
        {'id': 'simons', 'name': 'Simons Trader', 'style': 'Quantitative'},
        {'id': 'lynch', 'name': 'Lynch Trader', 'style': 'Growth Investing'},
        {'id': 'aggressive', 'name': 'Aggressive DQN Trader', 'style': 'High Risk'},
        {'id': 'conservative', 'name': 'Conservative DQN Trader', 'style': 'Low Risk'},
        {'id': 'momentum', 'name': 'Momentum DQN Trader', 'style': 'Trend Following'},
        {'id': 'contrarian', 'name': 'Contrarian DQN Trader', 'style': 'Counter-Trend'},
        {'id': 'swing', 'name': 'Swing DQN Trader', 'style': 'Medium-Term'},
        {'id': 'news', 'name': 'News Sentiment DQN Trader', 'style': 'News-Based'},
        {'id': 'scalper', 'name': 'Scalper DQN Trader', 'style': 'Short-Term'}
    ],
    'trader_portfolios': {
        'buffett': {
            'total_value': 18500.00,
            'base_currency': 'USDT',
            'base_amount': 3500.00,
            'assets_value': 15000.00,
            'num_assets': 2,
            'holdings': [
                {'symbol': 'BTC', 'amount': 0.25, 'value': 10000.00, 'price': 40000.00, 'change_24h': 2.5},
                {'symbol': 'ETH', 'amount': 4.0, 'value': 5000.00, 'price': 1250.00, 'change_24h': -1.2}
            ]
        },
        'soros': {
            'total_value': 16000.00,
            'base_currency': 'USDT',
            'base_amount': 6000.00,
            'assets_value': 10000.00,
            'num_assets': 3,
            'holdings': [
                {'symbol': 'BTC', 'amount': 0.15, 'value': 6000.00, 'price': 40000.00, 'change_24h': 2.5},
                {'symbol': 'SOL', 'amount': 30, 'value': 1500.00, 'price': 50.00, 'change_24h': 5.7},
                {'symbol': 'DOT', 'amount': 150, 'value': 2250.00, 'price': 15.00, 'change_24h': -0.5}
            ]
        },
        'simons': {
            'total_value': 14000.00,
            'base_currency': 'USDT',
            'base_amount': 4000.00,
            'assets_value': 10000.00,
            'num_assets': 4,
            'holdings': [
                {'symbol': 'BTC', 'amount': 0.10, 'value': 4000.00, 'price': 40000.00, 'change_24h': 2.5},
                {'symbol': 'ETH', 'amount': 2.0, 'value': 2500.00, 'price': 1250.00, 'change_24h': -1.2},
                {'symbol': 'SOL', 'amount': 20, 'value': 1000.00, 'price': 50.00, 'change_24h': 5.7},
                {'symbol': 'ADA', 'amount': 5000, 'value': 2500.00, 'price': 0.50, 'change_24h': 0.8}
            ]
        },
        'lynch': {
            'total_value': 17000.00,
            'base_currency': 'USDT',
            'base_amount': 5000.00,
            'assets_value': 12000.00,
            'num_assets': 3,
            'holdings': [
                {'symbol': 'BTC', 'amount': 0.20, 'value': 8000.00, 'price': 40000.00, 'change_24h': 2.5},
                {'symbol': 'ETH', 'amount': 1.5, 'value': 1800.00, 'price': 1200.00, 'change_24h': -1.2},
                {'symbol': 'SOL', 'amount': 40, 'value': 2000.00, 'price': 50.00, 'change_24h': 5.7}
            ]
        },
        'aggressive': {
            'total_value': 18000.00,
            'base_currency': 'USDT',
            'base_amount': 2000.00,
            'assets_value': 16000.00,
            'num_assets': 5,
            'holdings': [
                {'symbol': 'BTC', 'amount': 0.2, 'value': 8000.00, 'price': 40000.00, 'change_24h': 2.5},
                {'symbol': 'ETH', 'amount': 2.0, 'value': 2400.00, 'price': 1200.00, 'change_24h': -1.2},
                {'symbol': 'SOL', 'amount': 40, 'value': 2000.00, 'price': 50.00, 'change_24h': 5.7},
                {'symbol': 'DOT', 'amount': 100, 'value': 1500.00, 'price': 15.00, 'change_24h': -0.5},
                {'symbol': 'ADA', 'amount': 4200, 'value': 2100.00, 'price': 0.50, 'change_24h': 0.8}
            ]
        },
        'conservative': {
            'total_value': 15000.00,
            'base_currency': 'USDT',
            'base_amount': 8000.00,
            'assets_value': 7000.00,
            'num_assets': 2,
            'holdings': [
                {'symbol': 'BTC', 'amount': 0.125, 'value': 5000.00, 'price': 40000.00, 'change_24h': 2.5},
                {'symbol': 'ETH', 'amount': 1.67, 'value': 2000.00, 'price': 1200.00, 'change_24h': -1.2}
            ]
        }
    },
    'trade_history': {
        'buffett': [
            {'timestamp': '2023-04-01T10:22:33', 'action': 'buy', 'symbol': 'BTC', 'quantity': 0.125, 'price': 38000.00, 'value': 4750.00, 'fee': 4.75, 'status': 'completed'},
            {'timestamp': '2023-03-15T14:55:12', 'action': 'buy', 'symbol': 'ETH', 'quantity': 2.5, 'price': 1300.00, 'value': 3250.00, 'fee': 3.25, 'status': 'completed'},
            {'timestamp': '2023-03-10T09:30:45', 'action': 'buy', 'symbol': 'BTC', 'quantity': 0.125, 'price': 36000.00, 'value': 4500.00, 'fee': 4.50, 'status': 'completed'},
            {'timestamp': '2023-02-28T16:42:18', 'action': 'buy', 'symbol': 'ETH', 'quantity': 1.5, 'price': 1350.00, 'value': 2025.00, 'fee': 2.03, 'status': 'completed'}
        ],
        'soros': [
            {'timestamp': '2023-04-05T11:22:33', 'action': 'buy', 'symbol': 'BTC', 'quantity': 0.075, 'price': 39000.00, 'value': 2925.00, 'fee': 2.93, 'status': 'completed'},
            {'timestamp': '2023-04-02T10:15:22', 'action': 'buy', 'symbol': 'SOL', 'quantity': 20, 'price': 48.50, 'value': 970.00, 'fee': 0.97, 'status': 'completed'},
            {'timestamp': '2023-03-28T15:33:42', 'action': 'sell', 'symbol': 'ETH', 'quantity': 1.2, 'price': 1280.00, 'value': 1536.00, 'fee': 1.54, 'status': 'completed'},
            {'timestamp': '2023-03-25T09:45:11', 'action': 'buy', 'symbol': 'DOT', 'quantity': 75, 'price': 14.80, 'value': 1110.00, 'fee': 1.11, 'status': 'completed'},
            {'timestamp': '2023-03-20T14:22:36', 'action': 'buy', 'symbol': 'DOT', 'quantity': 75, 'price': 15.20, 'value': 1140.00, 'fee': 1.14, 'status': 'completed'},
            {'timestamp': '2023-03-15T08:55:23', 'action': 'buy', 'symbol': 'BTC', 'quantity': 0.075, 'price': 37500.00, 'value': 2812.50, 'fee': 2.81, 'status': 'completed'}
        ],
        'aggressive': [
            {'timestamp': '2023-04-08T08:12:45', 'action': 'buy', 'symbol': 'SOL', 'quantity': 20, 'price': 49.50, 'value': 990.00, 'fee': 0.99, 'status': 'completed'},
            {'timestamp': '2023-04-07T15:38:22', 'action': 'sell', 'symbol': 'DOT', 'quantity': 25, 'price': 15.20, 'value': 380.00, 'fee': 0.38, 'status': 'completed'},
            {'timestamp': '2023-04-05T10:25:33', 'action': 'buy', 'symbol': 'ADA', 'quantity': 2000, 'price': 0.48, 'value': 960.00, 'fee': 0.96, 'status': 'completed'},
            {'timestamp': '2023-04-03T14:18:52', 'action': 'buy', 'symbol': 'BTC', 'quantity': 0.05, 'price': 39500.00, 'value': 1975.00, 'fee': 1.98, 'status': 'completed'},
            {'timestamp': '2023-04-01T09:45:12', 'action': 'buy', 'symbol': 'ETH', 'quantity': 0.5, 'price': 1220.00, 'value': 610.00, 'fee': 0.61, 'status': 'completed'},
            {'timestamp': '2023-03-30T16:32:45', 'action': 'buy', 'symbol': 'DOT', 'quantity': 50, 'price': 14.90, 'value': 745.00, 'fee': 0.75, 'status': 'completed'},
            {'timestamp': '2023-03-28T11:22:18', 'action': 'sell', 'symbol': 'SOL', 'quantity': 10, 'price': 51.20, 'value': 512.00, 'fee': 0.51, 'status': 'completed'},
            {'timestamp': '2023-03-25T08:55:42', 'action': 'buy', 'symbol': 'ADA', 'quantity': 2200, 'price': 0.51, 'value': 1122.00, 'fee': 1.12, 'status': 'completed'}
        ]
    },
    'settings': {
        'general': {
            'baseCurrency': 'USDT',
            'defaultAmount': 100,
            'refreshInterval': 30,
            'darkMode': True
        },
        'traders': {
            'aggressive': {
                'enabled': True,
                'riskLevel': 8,
                'allocation': 20,
                'confidence': 60,
                'timeout': 7,
                'symbols': ['BTC', 'ETH', 'SOL', 'DOT']
            },
            'conservative': {
                'enabled': True,
                'riskLevel': 3,
                'allocation': 30,
                'confidence': 85,
                'timeout': 30,
                'symbols': ['BTC', 'ETH']
            }
        },
        'notifications': {
            'emailNotifications': True,
            'emailAddress': '',
            'pushNotifications': False,
            'triggers': {
                'tradeExecuted': True,
                'priceAlert': True,
                'portfolioChange': False,
                'traderError': True
            }
        },
        'api': {
            'exchange': 'binance',
            'apiKey': '',
            'apiSecret': '',
            'enableTrading': True
        }
    }
}

# Routes
@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio():
    """Render the portfolio page."""
    return render_template('portfolio.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# API endpoints
@app.route('/api/portfolio')
def get_portfolio():
    # In production, this would fetch data from your backend/database
    return jsonify(MOCK_DATA['portfolio'])

@app.route('/api/portfolio/<agent_id>')
def get_portfolio(agent_id):
    """Get portfolio data for an agent."""
    try:
        # Get current portfolio
        portfolio_history = db.get_portfolio_history(agent_id, limit=1)
        if not portfolio_history:
            return jsonify({"error": "No portfolio data found"}), 404
        
        current_portfolio = portfolio_history[0]
        
        # Get performance metrics
        performance_history = db.get_performance_history(agent_id, limit=1)
        performance = performance_history[0] if performance_history else {}
        
        # Get recent trades
        trade_history = db.get_trade_history(agent_id, limit=10)
        
        return jsonify({
            "portfolio": current_portfolio,
            "performance": performance,
            "recent_trades": trade_history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/traders')
def get_traders():
    # Return list of available traders
    return jsonify({'traders': MOCK_DATA['traders']})

@app.route('/api/prices')
def get_prices():
    # Return current prices
    return jsonify([
        {'symbol': symbol, 'price': data['price'], 'change24h': data['change_24h']}
        for symbol, data in MOCK_DATA['market_data'].items()
    ])

@app.route('/api/history/<trader_id>')
def get_trade_history(trader_id):
    # Get a specific trader's trade history
    if trader_id in MOCK_DATA['trade_history']:
        # Apply filters
        date_range = request.args.get('date_range', '30')
        action_filter = request.args.get('action', 'all')
        symbol_filter = request.args.get('symbol', 'all')
        min_amount = float(request.args.get('min_amount', '5'))
        page = int(request.args.get('page', '1'))
        
        # Filter trades
        filtered_trades = MOCK_DATA['trade_history'][trader_id]
        
        # Filter by date range
        if date_range != 'all':
            cutoff_date = datetime.now() - timedelta(days=int(date_range))
            filtered_trades = [
                trade for trade in filtered_trades 
                if datetime.fromisoformat(trade['timestamp']) > cutoff_date
            ]
        
        # Filter by action
        if action_filter != 'all':
            filtered_trades = [
                trade for trade in filtered_trades 
                if trade['action'] == action_filter
            ]
        
        # Filter by symbol
        if symbol_filter != 'all':
            filtered_trades = [
                trade for trade in filtered_trades 
                if trade['symbol'] == symbol_filter
            ]
        
        # Filter by minimum amount
        filtered_trades = [
            trade for trade in filtered_trades 
            if trade['value'] >= min_amount
        ]
        
        # Sort by date (newest first)
        filtered_trades.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Pagination
        page_size = 5
        total_pages = max(1, (len(filtered_trades) + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_trades))
        
        paged_trades = filtered_trades[start_idx:end_idx]
        
        return jsonify({
            'trades': paged_trades,
            'page': page,
            'total_pages': total_pages,
            'total_trades': len(filtered_trades)
        })
    else:
        # If trader doesn't have history yet, return empty array
        return jsonify({
            'trades': [],
            'page': 1,
            'total_pages': 1,
            'total_trades': 0
        })

@app.route('/api/settings/<section>', methods=['GET'])
def get_settings(section):
    # Get settings for a specific section
    if section in MOCK_DATA['settings']:
        return jsonify(MOCK_DATA['settings'][section])
    else:
        return jsonify({'error': 'Settings section not found'}), 404

@app.route('/api/settings/<section>', methods=['POST'])
def update_settings(section):
    # Update settings for a specific section
    if section in MOCK_DATA['settings']:
        data = request.json
        MOCK_DATA['settings'][section].update(data)
        return jsonify({'success': True, 'message': f'{section.capitalize()} settings saved successfully'})
    else:
        return jsonify({'error': 'Settings section not found'}), 404

@app.route('/api/trade', methods=['POST'])
def execute_trade():
    data = request.json
    action = data.get('action')  # 'buy' or 'sell'
    symbol = data.get('symbol')
    amount = float(data.get('amount', 0))
    
    if not symbol or amount <= 0:
        return jsonify({'success': False, 'message': 'Invalid trade parameters'})
    
    # Check minimum trade value ($5)
    price = MOCK_DATA['market_data'].get(symbol, {}).get('price', 0)
    trade_value = price * amount
    
    if trade_value < 5:
        return jsonify({'success': False, 'message': f'Trade value must be at least $5. Current value: ${trade_value:.2f}'})
    
    # In a real system, this would execute the trade through your backend
    # For now, we'll just update our mock data
    if action == 'buy':
        # Check if we have enough base currency
        if MOCK_DATA['portfolio']['base_amount'] < trade_value:
            return jsonify({'success': False, 'message': 'Insufficient funds'})
        
        # Update portfolio
        MOCK_DATA['portfolio']['base_amount'] -= trade_value
        
        # Check if we already own this asset
        for holding in MOCK_DATA['portfolio']['holdings']:
            if holding['symbol'] == symbol:
                holding['amount'] += amount
                holding['value'] = holding['amount'] * price
                break
        else:
            # Add new holding
            MOCK_DATA['portfolio']['holdings'].append({
                'symbol': symbol,
                'amount': amount,
                'value': amount * price,
                'price': price,
                'change_24h': MOCK_DATA['market_data'][symbol]['change_24h']
            })
            MOCK_DATA['portfolio']['num_assets'] += 1
        
    elif action == 'sell':
        # Check if we own this asset and have enough of it
        for i, holding in enumerate(MOCK_DATA['portfolio']['holdings']):
            if holding['symbol'] == symbol:
                if holding['amount'] < amount:
                    return jsonify({'success': False, 'message': 'Insufficient assets to sell'})
                
                # Update holding
                holding['amount'] -= amount
                holding['value'] = holding['amount'] * price
                
                # Add funds to base currency
                MOCK_DATA['portfolio']['base_amount'] += trade_value
                
                # Remove holding if amount is 0
                if holding['amount'] <= 0:
                    MOCK_DATA['portfolio']['holdings'].pop(i)
                    MOCK_DATA['portfolio']['num_assets'] -= 1
                
                break
        else:
            return jsonify({'success': False, 'message': f'You do not own any {symbol}'})
    
    # Update total values
    assets_value = sum(h['value'] for h in MOCK_DATA['portfolio']['holdings'])
    MOCK_DATA['portfolio']['assets_value'] = assets_value
    MOCK_DATA['portfolio']['total_value'] = MOCK_DATA['portfolio']['base_amount'] + assets_value
    
    return jsonify({
        'success': True, 
        'message': f'Successfully {"bought" if action == "buy" else "sold"} {amount} {symbol}',
        'portfolio': MOCK_DATA['portfolio']
    })

@app.route('/api/buy', methods=['POST'])
def buy():
    data = request.json
    trader_id = data.get('trader_id')
    symbol = data.get('symbol')
    amount = float(data.get('amount', 0))
    
    if not trader_id or not symbol or amount <= 0:
        return jsonify({'success': False, 'message': 'Invalid parameters'}), 400
    
    if trader_id not in MOCK_DATA['trader_portfolios']:
        return jsonify({'success': False, 'message': 'Trader not found'}), 404
    
    # Get the trader's portfolio
    portfolio = MOCK_DATA['trader_portfolios'][trader_id]
    
    # Check minimum trade value ($5)
    if amount < 5:
        return jsonify({'success': False, 'message': 'Minimum trade amount is $5'}), 400
    
    # Check if we have enough base currency
    if portfolio['base_amount'] < amount:
        return jsonify({'success': False, 'message': 'Insufficient funds'}), 400
    
    # Get current price
    price = MOCK_DATA['market_data'].get(symbol, {}).get('price', 0)
    if price <= 0:
        return jsonify({'success': False, 'message': 'Invalid symbol or price'}), 400
    
    # Calculate quantity
    quantity = amount / price
    
    # Update portfolio
    portfolio['base_amount'] -= amount
    
    # Check if we already own this asset
    for holding in portfolio['holdings']:
        if holding['symbol'] == symbol:
            holding['amount'] += quantity
            holding['value'] = holding['amount'] * price
            break
    else:
        # Add new holding
        portfolio['holdings'].append({
            'symbol': symbol,
            'amount': quantity,
            'value': quantity * price,
            'price': price,
            'change_24h': MOCK_DATA['market_data'][symbol]['change_24h']
        })
        portfolio['num_assets'] += 1
    
    # Update total values
    assets_value = sum(h['value'] for h in portfolio['holdings'])
    portfolio['assets_value'] = assets_value
    portfolio['total_value'] = portfolio['base_amount'] + assets_value
    
    # Add trade to history
    if trader_id not in MOCK_DATA['trade_history']:
        MOCK_DATA['trade_history'][trader_id] = []
    
    MOCK_DATA['trade_history'][trader_id].insert(0, {
        'timestamp': datetime.now().isoformat(),
        'action': 'buy',
        'symbol': symbol,
        'quantity': quantity,
        'price': price,
        'value': amount,
        'fee': amount * 0.001,  # 0.1% fee
        'status': 'completed'
    })
    
    return jsonify({
        'success': True, 
        'message': f'Successfully bought {quantity:.8f} {symbol} for ${amount:.2f}',
        'data': {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'amount': amount
        }
    })

@app.route('/api/sell', methods=['POST'])
def sell():
    data = request.json
    trader_id = data.get('trader_id')
    symbol = data.get('symbol')
    amount = float(data.get('amount', 0))
    
    if not trader_id or not symbol or amount <= 0:
        return jsonify({'success': False, 'message': 'Invalid parameters'}), 400
    
    if trader_id not in MOCK_DATA['trader_portfolios']:
        return jsonify({'success': False, 'message': 'Trader not found'}), 404
    
    # Get the trader's portfolio
    portfolio = MOCK_DATA['trader_portfolios'][trader_id]
    
    # Check minimum trade value ($5)
    if amount < 5:
        return jsonify({'success': False, 'message': 'Minimum trade amount is $5'}), 400
    
    # Get current price
    price = MOCK_DATA['market_data'].get(symbol, {}).get('price', 0)
    if price <= 0:
        return jsonify({'success': False, 'message': 'Invalid symbol or price'}), 400
    
    # Calculate quantity
    quantity = amount / price
    
    # Check if we own this asset and have enough of it
    for i, holding in enumerate(portfolio['holdings']):
        if holding['symbol'] == symbol:
            if holding['amount'] < quantity:
                return jsonify({'success': False, 'message': 'Insufficient assets to sell'}), 400
            
            # Update holding
            holding['amount'] -= quantity
            holding['value'] = holding['amount'] * price
            
            # Add funds to base currency
            portfolio['base_amount'] += amount
            
            # Remove holding if amount is 0
            if holding['amount'] <= 0:
                portfolio['holdings'].pop(i)
                portfolio['num_assets'] -= 1
            
            # Update total values
            assets_value = sum(h['value'] for h in portfolio['holdings'])
            portfolio['assets_value'] = assets_value
            portfolio['total_value'] = portfolio['base_amount'] + assets_value
            
            # Add trade to history
            if trader_id not in MOCK_DATA['trade_history']:
                MOCK_DATA['trade_history'][trader_id] = []
            
            MOCK_DATA['trade_history'][trader_id].insert(0, {
                'timestamp': datetime.now().isoformat(),
                'action': 'sell',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'value': amount,
                'fee': amount * 0.001,  # 0.1% fee
                'status': 'completed'
            })
            
            return jsonify({
                'success': True, 
                'message': f'Successfully sold {quantity:.8f} {symbol} for ${amount:.2f}',
                'data': {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'amount': amount
                }
            })
    
    return jsonify({'success': False, 'message': f'You do not own any {symbol}'}), 400

@app.route('/api/trades/<agent_id>')
def get_trades(agent_id):
    """Get trade history for an agent."""
    try:
        limit = request.args.get('limit', default=100, type=int)
        trades = db.get_trade_history(agent_id, limit)
        return jsonify(trades)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/<agent_id>')
def get_performance(agent_id):
    """Get performance metrics for an agent."""
    try:
        limit = request.args.get('limit', default=100, type=int)
        metrics = db.get_performance_history(agent_id, limit)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/history/<agent_id>')
def get_portfolio_history(agent_id):
    """Get portfolio history for an agent."""
    try:
        symbol = request.args.get('symbol')
        limit = request.args.get('limit', default=100, type=int)
        history = db.get_portfolio_history(agent_id, symbol, limit)
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_portfolio_history(current_value):
    """Generate mock portfolio history data for the last 30 days"""
    history = []
    base_value = current_value * 0.7  # Start at 70% of current value
    
    # Generate daily values with some randomness
    for i in range(30):
        date = (datetime.now() - timedelta(days=29-i)).strftime('%Y-%m-%d')
        # Progressive growth with randomness
        value = base_value + (current_value - base_value) * (i/29) + random.uniform(-500, 500)
        history.append({'date': date, 'value': value})
    
    return history

if __name__ == '__main__':
    socketio.run(app, debug=True) 