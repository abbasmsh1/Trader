from main import TradingSystem, create_dashboard
from futures_trading_system import FuturesTradingSystem, create_futures_dashboard
from threading import Thread
import webbrowser
import time

def main():
    # Create spot trading system
    spot_system = TradingSystem()
    
    # Create futures trading system with 3x leverage
    futures_system = FuturesTradingSystem(leverage=3.0)
    
    # Start spot trading system in a background thread
    spot_thread = Thread(target=spot_system.run)
    spot_thread.daemon = True
    spot_thread.start()
    
    # Start futures trading system in a background thread
    futures_thread = Thread(target=futures_system.run)
    futures_thread.daemon = True
    futures_thread.start()
    
    # Create and run both dashboards
    spot_app = create_dashboard(spot_system)
    futures_app = create_futures_dashboard(futures_system)
    
    # Open browsers for both dashboards
    webbrowser.open('http://127.0.0.1:8050')  # Spot trading
    webbrowser.open('http://127.0.0.1:8051/futures/')  # Futures trading
    
    # Run the spot dashboard on port 8050
    spot_server = Thread(target=lambda: spot_app.run_server(debug=False, port=8050))
    spot_server.daemon = True
    spot_server.start()
    
    # Run the futures dashboard on port 8051
    futures_app.run_server(debug=False, port=8051)

if __name__ == '__main__':
    main() 