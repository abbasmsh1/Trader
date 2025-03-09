import os
import sys
from threading import Thread
from main import TradingSystem, create_dashboard

def main():
    """Run the trading system with a single dashboard."""
    # Create trading system instance
    trading_system = TradingSystem()
    
    # Initialize the system
    trading_system._load_state()  # Load previous state if available
    
    # Start the trading system in a background thread
    trading_thread = Thread(target=trading_system.run)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Create and run the dashboard
    app = create_dashboard(trading_system)
    print("Dashboard running on http://0.0.0.0:8050/")
    app.run_server(host='0.0.0.0', port=8050, debug=False)

if __name__ == "__main__":
    main() 