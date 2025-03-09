import os
from main import TradingSystem, create_dashboard
import dash
from dash import dcc, html
from threading import Thread

def main():
    # Initialize trading system
    ts = TradingSystem()
    
    # Load previous state if available
    ts._load_state()
    
    # Create Dash app with explicit assets folder
    assets_path = os.path.join(os.path.dirname(__file__), 'assets')
    
    # Create the dashboard with assets configuration
    app = create_dashboard(ts)
    
    # Configure scripts to serve locally
    app.scripts.config.serve_locally = True
    app.css.config.serve_locally = True
    
    # Start trading system in background thread
    trading_thread = Thread(target=ts.run)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Run dashboard with development mode disabled
    print("Starting dashboard server...")
    app.run_server(
        host='0.0.0.0',
        port=8050,
        debug=False,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_serve_dev_bundles=False
    )

if __name__ == '__main__':
    main() 