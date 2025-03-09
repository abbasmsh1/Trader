import os
import sys
from threading import Thread
import dash
from main import TradingSystem, create_dashboard

def main():
    # Create two separate trading systems
    usd_ts = TradingSystem(goal_filter='usd')
    btc_ts = TradingSystem(goal_filter='btc')
    
    # Create USD goal dashboard
    usd_app = create_dashboard(usd_ts, title="USD Goal Trading Arena", subtitle="AI Traders Battle: $20 → $100 Challenge")
    
    # Create BTC goal dashboard
    btc_app = create_dashboard(btc_ts, title="BTC Goal Trading Arena", subtitle="AI Traders Battle: 1 BTC Accumulation Challenge")
    
    # Start trading systems in background threads
    usd_thread = Thread(target=usd_ts.run, daemon=True)
    btc_thread = Thread(target=btc_ts.run, daemon=True)
    
    usd_thread.start()
    btc_thread.start()
    
    # Run USD dashboard on port 8050
    print("USD Goal Dashboard running on http://0.0.0.0:8050/")
    
    # Run BTC dashboard on port 8060
    print("BTC Goal Dashboard running on http://0.0.0.0:8060/")
    
    # Start the BTC dashboard in a separate thread
    btc_dash_thread = Thread(target=lambda: btc_app.run_server(host='0.0.0.0', port=8060, debug=False), daemon=True)
    btc_dash_thread.start()
    
    # Run the USD dashboard in the main thread
    usd_app.run_server(host='0.0.0.0', port=8050, debug=False)

if __name__ == "__main__":
    main() 