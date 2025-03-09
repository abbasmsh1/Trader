from main import TradingSystem, run_dashboard
from futures_trading_system import FuturesTradingSystem, run_futures_dashboard
import threading

def main():
    # Create trading systems
    spot_system = TradingSystem()
    futures_system = FuturesTradingSystem(leverage=3.0)
    
    # Start trading systems in background threads
    spot_thread = threading.Thread(target=spot_system.run, daemon=True)
    futures_thread = threading.Thread(target=futures_system.run, daemon=True)
    
    spot_thread.start()
    futures_thread.start()
    
    # Create and run dashboards in separate threads
    spot_dashboard_thread = threading.Thread(target=lambda: run_dashboard(spot_system), daemon=True)
    futures_dashboard_thread = threading.Thread(target=lambda: run_futures_dashboard(futures_system), daemon=True)
    
    spot_dashboard_thread.start()
    futures_dashboard_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            spot_thread.join(1)
            futures_thread.join(1)
            spot_dashboard_thread.join(1)
            futures_dashboard_thread.join(1)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")

if __name__ == '__main__':
    main() 