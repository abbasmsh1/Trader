#!/usr/bin/env python
"""
Wallet Demo Script

This script demonstrates the wallet functionality by simulating a simple
trading strategy with BTC and ETH over a period of time.
"""

import logging
import time
import random
import os
import json
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wallet_demo")

# Import wallet
from models.wallet import Wallet

class WalletDemo:
    """Simple demonstration of wallet functionality with simulated trading."""
    
    def __init__(self, initial_balance=10000.0, days=30, volatility=0.03):
        """
        Initialize the wallet demo.
        
        Args:
            initial_balance: Starting USDT balance
            days: Number of days to simulate
            volatility: Daily price volatility factor
        """
        self.initial_balance = initial_balance
        self.days = days
        self.volatility = volatility
        
        # Create wallet
        self.wallet = Wallet(
            initial_balance=initial_balance,
            base_currency="USDT",
            name="Demo Trading Wallet"
        )
        
        # Initialize starting prices
        self.prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "SOL": 100.0,
            "BNB": 400.0
        }
        
        # Track historical prices and portfolio values
        self.price_history = {symbol: [] for symbol in self.prices}
        self.portfolio_history = []
        
        # Create output directory
        os.makedirs("data", exist_ok=True)
        
        logger.info(f"Initialized wallet demo with {initial_balance} USDT for {days} days")
    
    def simulate(self):
        """Run the trading simulation."""
        logger.info("Starting trading simulation")
        
        # Initial asset distribution - buying some BTC and ETH
        self._initial_allocation()
        
        # Save initial snapshot
        self._save_portfolio_snapshot()
        
        # Run daily simulation
        for day in range(1, self.days + 1):
            # Update prices
            self._update_market_prices(day)
            
            # Execute daily trading strategy
            self._execute_trading_strategy(day)
            
            # Save daily snapshot
            self._save_portfolio_snapshot()
            
            # Log progress
            if day % 5 == 0 or day == 1:
                self._log_current_status(day)
        
        # Save results
        self._save_results()
        
        # Create charts
        self._create_charts()
        
        # Final report
        self._generate_final_report()
    
    def _initial_allocation(self):
        """Perform initial asset allocation."""
        # Buy some BTC
        btc_allocation = self.initial_balance * 0.3
        btc_price = self.prices["BTC"]
        btc_amount = btc_allocation / btc_price
        
        self.wallet.add_trade(
            trade_type="buy",
            from_currency="USDT",
            to_currency="BTC",
            from_amount=btc_allocation,
            to_amount=btc_amount,
            price=btc_price,
            fee=btc_allocation * 0.001,
            exchange="demo"
        )
        
        # Buy some ETH
        eth_allocation = self.initial_balance * 0.2
        eth_price = self.prices["ETH"]
        eth_amount = eth_allocation / eth_price
        
        self.wallet.add_trade(
            trade_type="buy",
            from_currency="USDT",
            to_currency="ETH",
            from_amount=eth_allocation,
            to_amount=eth_amount,
            price=eth_price,
            fee=eth_allocation * 0.001,
            exchange="demo"
        )
        
        logger.info(f"Initial allocation: {btc_amount:.4f} BTC @ {btc_price} and {eth_amount:.4f} ETH @ {eth_price}")
    
    def _update_market_prices(self, day):
        """Update market prices based on simulated market movement."""
        # Apply random price movement with some trend bias
        # Later days have higher chance of uptrend
        trend_bias = 0.0
        if day > self.days / 2:
            trend_bias = 0.01  # Slight upward bias in later half
        
        for symbol in self.prices:
            # Calculate daily price change with random walk
            daily_change = random.normalvariate(trend_bias, self.volatility)
            
            # Apply change to price
            self.prices[symbol] *= (1 + daily_change)
            
            # Record price in history
            self.price_history[symbol].append(self.prices[symbol])
    
    def _execute_trading_strategy(self, day):
        """Execute a simple trading strategy."""
        # Get current balances
        balances = self.wallet.get_all_balances()
        
        # Strategy decisions
        if day % 7 == 1:  # Every 7 days, rebalance portfolio
            self._rebalance_portfolio()
        elif day % 5 == 0:  # Every 5 days, rotate some assets
            self._rotate_assets()
        elif day % 3 == 0:  # Every 3 days, take some profits if available
            self._take_profits()
    
    def _rebalance_portfolio(self):
        """Rebalance portfolio to target allocations."""
        logger.info("Rebalancing portfolio")
        
        # Target allocations
        targets = {
            "BTC": 0.3,
            "ETH": 0.2,
            "SOL": 0.1,
            "BNB": 0.1,
            "USDT": 0.3
        }
        
        # Get current portfolio valuation
        valuation = self.wallet.calculate_total_value(self.prices)
        total_value = valuation["total_value"]
        
        # Current allocations
        current = {
            "BTC": (self.wallet.get_balance("BTC") * self.prices["BTC"]) / total_value,
            "ETH": (self.wallet.get_balance("ETH") * self.prices["ETH"]) / total_value,
            "SOL": (self.wallet.get_balance("SOL") * self.prices["SOL"]) / total_value,
            "BNB": (self.wallet.get_balance("BNB") * self.prices["BNB"]) / total_value,
            "USDT": self.wallet.get_balance("USDT") / total_value
        }
        
        # Calculate differences and execute trades to rebalance
        for symbol, target_pct in targets.items():
            if symbol == "USDT":
                continue  # Skip USDT, it will be the result of other trades
                
            current_pct = current.get(symbol, 0)
            diff_pct = target_pct - current_pct
            
            if abs(diff_pct) < 0.02:
                continue  # Ignore small differences
                
            if diff_pct > 0:
                # Need to buy more
                amount_to_buy_usdt = diff_pct * total_value
                if amount_to_buy_usdt > self.wallet.get_balance("USDT"):
                    amount_to_buy_usdt = self.wallet.get_balance("USDT") * 0.9  # Keep some reserves
                
                if amount_to_buy_usdt > 10:  # Minimum order size
                    self._execute_buy(symbol, amount_to_buy_usdt)
            else:
                # Need to sell some
                current_amount = self.wallet.get_balance(symbol)
                amount_to_sell = current_amount * abs(diff_pct) / current_pct
                
                if amount_to_sell * self.prices[symbol] > 10:  # Minimum order size
                    self._execute_sell(symbol, amount_to_sell)
    
    def _rotate_assets(self):
        """Rotate between assets based on simple momentum."""
        logger.info("Rotating assets based on momentum")
        
        # Calculate 3-day price changes
        momentum = {}
        for symbol in ["BTC", "ETH", "SOL", "BNB"]:
            if len(self.price_history[symbol]) >= 3:
                price_change = self.price_history[symbol][-1] / self.price_history[symbol][-3] - 1
                momentum[symbol] = price_change
        
        if not momentum:
            return
            
        # Find best and worst performing assets
        best_symbol = max(momentum.items(), key=lambda x: x[1])[0]
        worst_symbol = min(momentum.items(), key=lambda x: x[1])[0]
        
        # Sell some of worst performer if we have it
        worst_balance = self.wallet.get_balance(worst_symbol)
        if worst_balance > 0 and momentum[worst_symbol] < -0.02:  # If it dropped by more than 2%
            amount_to_sell = worst_balance * 0.5  # Sell half
            if amount_to_sell * self.prices[worst_symbol] > 10:  # Minimum order size
                self._execute_sell(worst_symbol, amount_to_sell)
                
                # Use proceeds to buy best performer
                usdt_available = self.wallet.get_balance("USDT")
                if usdt_available > 100 and momentum[best_symbol] > 0.02:  # If it gained more than 2%
                    self._execute_buy(best_symbol, usdt_available * 0.8)
    
    def _take_profits(self):
        """Take profits from positions that have gained significantly."""
        logger.info("Checking for profit-taking opportunities")
        
        # Get trade history to check for profitable positions
        trades = self.wallet.get_trade_history(trade_type="buy", limit=20)
        
        for trade in trades:
            symbol = trade["to_currency"]
            entry_price = trade["price"]
            current_price = self.prices.get(symbol)
            
            if not current_price:
                continue
                
            # Calculate profit percentage
            profit_pct = (current_price / entry_price - 1) * 100
            
            # Take profits if gained more than 10%
            if profit_pct > 10:
                balance = self.wallet.get_balance(symbol)
                if balance > 0:
                    # Sell 50% of position
                    amount_to_sell = balance * 0.5
                    if amount_to_sell * current_price > 10:  # Minimum order size
                        logger.info(f"Taking profits on {symbol}: +{profit_pct:.2f}%")
                        self._execute_sell(symbol, amount_to_sell)
    
    def _execute_buy(self, symbol, usdt_amount):
        """Execute a buy trade for the given symbol and USDT amount."""
        price = self.prices[symbol]
        amount = usdt_amount / price
        
        try:
            self.wallet.add_trade(
                trade_type="buy",
                from_currency="USDT",
                to_currency=symbol,
                from_amount=usdt_amount,
                to_amount=amount,
                price=price,
                fee=usdt_amount * 0.001,
                exchange="demo"
            )
            logger.info(f"Bought {amount:.4f} {symbol} @ {price:.2f} for {usdt_amount:.2f} USDT")
            return True
        except ValueError as e:
            logger.warning(f"Buy failed: {str(e)}")
            return False
    
    def _execute_sell(self, symbol, amount):
        """Execute a sell trade for the given symbol and amount."""
        price = self.prices[symbol]
        usdt_amount = amount * price
        fee = usdt_amount * 0.001
        
        try:
            self.wallet.add_trade(
                trade_type="sell",
                from_currency=symbol,
                to_currency="USDT",
                from_amount=amount,
                to_amount=usdt_amount - fee,
                price=price,
                fee=fee,
                exchange="demo"
            )
            logger.info(f"Sold {amount:.4f} {symbol} @ {price:.2f} for {usdt_amount:.2f} USDT")
            return True
        except ValueError as e:
            logger.warning(f"Sell failed: {str(e)}")
            return False
    
    def _save_portfolio_snapshot(self):
        """Save a snapshot of current portfolio value and allocation."""
        valuation = self.wallet.calculate_total_value(self.prices)
        self.portfolio_history.append({
            "timestamp": datetime.now().isoformat(),
            "total_value": valuation["total_value"],
            "balances": self.wallet.get_all_balances(),
            "prices": self.prices.copy()
        })
    
    def _log_current_status(self, day):
        """Log current portfolio status."""
        valuation = self.wallet.calculate_total_value(self.prices)
        
        logger.info(f"Day {day}: Portfolio value: {valuation['total_value']:.2f} USDT")
        logger.info(f"  BTC: {self.prices['BTC']:.2f}, ETH: {self.prices['ETH']:.2f}, " +
                    f"SOL: {self.prices['SOL']:.2f}, BNB: {self.prices['BNB']:.2f}")
        
        # Log holdings
        for currency, balance in self.wallet.get_all_balances().items():
            if balance > 0:
                if currency == "USDT":
                    logger.info(f"  {currency}: {balance:.2f}")
                else:
                    value = balance * self.prices.get(currency, 0)
                    logger.info(f"  {currency}: {balance:.4f} (Value: {value:.2f} USDT)")
    
    def _save_results(self):
        """Save simulation results to file."""
        results = {
            "initial_balance": self.initial_balance,
            "days_simulated": self.days,
            "final_value": self.portfolio_history[-1]["total_value"],
            "return_pct": (self.portfolio_history[-1]["total_value"] / self.initial_balance - 1) * 100,
            "price_history": self.price_history,
            "portfolio_history": self.portfolio_history,
            "final_balances": self.wallet.get_all_balances(),
            "trades": self.wallet.get_trade_history(limit=1000)
        }
        
        with open("data/wallet_demo_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to data/wallet_demo_results.json")
    
    def _create_charts(self):
        """Create charts of portfolio performance and asset prices."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Extract data for plotting
            dates = [i+1 for i in range(len(self.portfolio_history))]
            portfolio_values = [p["total_value"] for p in self.portfolio_history]
            
            # Create portfolio value chart
            plt.figure(figsize=(12, 6))
            plt.plot(dates, portfolio_values, 'b-', linewidth=2)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Day')
            plt.ylabel('Value (USDT)')
            plt.grid(True)
            plt.savefig('data/portfolio_value.png')
            logger.info("Created portfolio value chart: data/portfolio_value.png")
            
            # Create asset price chart
            plt.figure(figsize=(12, 6))
            
            # Normalize prices to starting value for comparison
            for symbol, history in self.price_history.items():
                normalized = [price / history[0] for price in history]
                plt.plot(range(1, len(normalized)+1), normalized, linewidth=2, label=symbol)
            
            plt.title('Asset Price Performance (Normalized)')
            plt.xlabel('Day')
            plt.ylabel('Price (Normalized)')
            plt.legend()
            plt.grid(True)
            plt.savefig('data/asset_prices.png')
            logger.info("Created asset price chart: data/asset_prices.png")
            
            # Create allocation pie chart for final day
            final_allocation = {}
            for currency, balance in self.wallet.get_all_balances().items():
                if balance > 0:
                    if currency == "USDT":
                        final_allocation[currency] = balance
                    else:
                        value = balance * self.prices.get(currency, 0)
                        final_allocation[currency] = value
            
            # Only include non-zero allocations
            labels = [k for k, v in final_allocation.items() if v > 0]
            sizes = [v for k, v in final_allocation.items() if v > 0]
            
            plt.figure(figsize=(10, 10))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Final Portfolio Allocation')
            plt.savefig('data/final_allocation.png')
            logger.info("Created allocation chart: data/final_allocation.png")
            
        except Exception as e:
            logger.error(f"Error creating charts: {str(e)}")
    
    def _generate_final_report(self):
        """Generate a final report of the simulation."""
        # Calculate key metrics
        initial_value = self.initial_balance
        final_value = self.portfolio_history[-1]["total_value"]
        total_return = final_value - initial_value
        total_return_pct = (final_value / initial_value - 1) * 100
        
        # Calculate max drawdown
        peak = initial_value
        max_drawdown = 0
        
        for snapshot in self.portfolio_history:
            value = snapshot["total_value"]
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Get trade statistics
        trades = self.wallet.get_trade_history(limit=1000)
        buy_trades = [t for t in trades if t["type"] == "buy"]
        sell_trades = [t for t in trades if t["type"] == "sell"]
        
        # Print report
        report = f"""
        ===== WALLET DEMO SIMULATION REPORT =====
        
        Simulation Parameters:
          - Initial Balance: {initial_value:.2f} USDT
          - Days Simulated: {self.days}
          - Volatility Factor: {self.volatility}
        
        Performance:
          - Final Portfolio Value: {final_value:.2f} USDT
          - Total Return: {total_return:.2f} USDT ({total_return_pct:.2f}%)
          - Max Drawdown: {max_drawdown:.2f}%
        
        Trading Activity:
          - Total Trades: {len(trades)}
          - Buy Trades: {len(buy_trades)}
          - Sell Trades: {len(sell_trades)}
        
        Final Balances:
        """
        
        for currency, balance in self.wallet.get_all_balances().items():
            if balance > 0:
                if currency == "USDT":
                    report += f"  - {currency}: {balance:.2f}\n"
                else:
                    value = balance * self.prices.get(currency, 0)
                    report += f"  - {currency}: {balance:.4f} (Value: {value:.2f} USDT)\n"
        
        report += "\nCharts created in the data/ directory"
        
        print(report)
        
        # Also save to file
        with open("data/wallet_demo_report.txt", "w") as f:
            f.write(report)
        
        logger.info("Final report generated: data/wallet_demo_report.txt")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Wallet Demo Simulation")
    parser.add_argument("--initial", type=float, default=10000.0, help="Initial balance in USDT")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--volatility", type=float, default=0.03, help="Daily price volatility factor")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Run the demo
    demo = WalletDemo(
        initial_balance=args.initial,
        days=args.days,
        volatility=args.volatility
    )
    
    demo.simulate() 