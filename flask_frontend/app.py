from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import json
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Routes
@app.route('/')
def index():
    """Render the dashboard page."""
    return render_template('index.html')

@app.route('/traders')
def traders():
    """Render the traders page."""
    return render_template('traders.html')

@app.route('/trader/<trader_id>')
def trader_detail(trader_id):
    """Render the trader detail page."""
    return render_template('trader_detail.html', trader_id=trader_id)

@app.route('/market')
def market():
    """Render the market page."""
    return render_template('market.html')

@app.route('/communications')
def communications():
    """Render the communications page."""
    return render_template('communications.html')

# API proxy routes
@app.route('/api/traders')
def get_traders():
    """Proxy to backend API for traders data."""
    response = requests.get(f"{BACKEND_URL}/traders")
    return jsonify(response.json())

@app.route('/api/traders/<trader_id>')
def get_trader(trader_id):
    """Proxy to backend API for specific trader data."""
    response = requests.get(f"{BACKEND_URL}/traders/{trader_id}")
    return jsonify(response.json())

@app.route('/api/market')
def get_market():
    """Proxy to backend API for market data."""
    response = requests.get(f"{BACKEND_URL}/market")
    return jsonify(response.json())

@app.route('/api/market/<trading_pair>')
def get_pair_data(trading_pair):
    """Proxy to backend API for specific trading pair data."""
    response = requests.get(f"{BACKEND_URL}/market/{trading_pair}")
    return jsonify(response.json())

@app.route('/api/communications')
def get_communications():
    """Proxy to backend API for communications data."""
    response = requests.get(f"{BACKEND_URL}/communications")
    return jsonify(response.json())

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(host='0.0.0.0', port=3001, debug=True) 