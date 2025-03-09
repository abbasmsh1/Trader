import argparse
import threading
from trader.core.trading_system import TradingSystem
from trader.ui.dashboard import create_dashboard
from trader.agents.value_investor import ValueInvestor
from trader.agents.tech_disruptor import TechDisruptor
from trader.agents.trend_follower import TrendFollower
from trader.agents.contrarian_trader import ContrarianTrader
from trader.agents.macro_trader import MacroTrader
from trader.agents.swing_trader import SwingTrader

def create_agents(goal_filter=None):
    """
    Create a set of trading agents.
    
    Args:
        goal_filter: Filter agents by goal type ('usd' or 'btc')
        
    Returns:
        List of trading agents
    """
    agents = [
        ValueInvestor(name="Warren Buffett AI", timeframe='1d'),
        TechDisruptor(name="Elon Musk AI", timeframe='1h'),
        TrendFollower(name="Michael Burry AI", timeframe='1d'),
        ContrarianTrader(name="Ray Dalio AI", timeframe='1d'),
        MacroTrader(name="George Soros AI", timeframe='1d'),
        SwingTrader(name="Jesse Livermore AI", timeframe='4h')
    ]
    
    return agents

def run_single_dashboard():
    """Run the trading system with a single dashboard."""
    # Create agents
    agents = create_agents()
    
    # Create trading system
    trading_system = TradingSystem(agents=agents)
    
    # Start the trading system in a background thread
    trading_thread = threading.Thread(target=trading_system.run, daemon=True)
    trading_thread.start()
    
    # Create and run the dashboard
    app = create_dashboard(trading_system)
    print("Dashboard running on http://0.0.0.0:8050/")
    app.run_server(host='0.0.0.0', port=8050, debug=False)

def run_dual_dashboard():
    """Run the trading system with dual dashboards (USD and BTC goals)."""
    # Create two separate trading systems
    usd_agents = create_agents(goal_filter='usd')
    btc_agents = create_agents(goal_filter='btc')
    
    usd_ts = TradingSystem(agents=usd_agents, goal_filter='usd')
    btc_ts = TradingSystem(agents=btc_agents, goal_filter='btc')
    
    # Create USD goal dashboard
    usd_app = create_dashboard(usd_ts, title="USD Goal Trading Arena", subtitle="AI Traders Battle: $20 → $100 Challenge")
    
    # Create BTC goal dashboard
    btc_app = create_dashboard(btc_ts, title="BTC Goal Trading Arena", subtitle="AI Traders Battle: 1 BTC Accumulation Challenge")
    
    # Start trading systems in background threads
    usd_thread = threading.Thread(target=usd_ts.run, daemon=True)
    btc_thread = threading.Thread(target=btc_ts.run, daemon=True)
    
    usd_thread.start()
    btc_thread.start()
    
    # Run USD dashboard on port 8050
    print("USD Goal Dashboard running on http://0.0.0.0:8050/")
    
    # Run BTC dashboard on port 8060
    print("BTC Goal Dashboard running on http://0.0.0.0:8060/")
    
    # Start the BTC dashboard in a separate thread
    btc_dash_thread = threading.Thread(target=lambda: btc_app.run_server(host='0.0.0.0', port=8060, debug=False), daemon=True)
    btc_dash_thread.start()
    
    # Run the USD dashboard in the main thread
    usd_app.run_server(host='0.0.0.0', port=8050, debug=False)

def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(description='Run the AI Trader Dashboard')
    parser.add_argument('--mode', choices=['single', 'dual'], default='single',
                        help='Dashboard mode: single (default) or dual (USD and BTC goals)')
    
    args = parser.parse_args()
    
    if args.mode == 'dual':
        run_dual_dashboard()
    else:
        run_single_dashboard()

if __name__ == "__main__":
    main() 