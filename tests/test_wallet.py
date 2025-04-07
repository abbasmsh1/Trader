#!/usr/bin/env python
"""
Wallet Test Script

This script tests the functionality of the Wallet class,
performing basic operations like deposits, trades, and portfolio valuation.
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wallet_test")

# Import wallet
from models.wallet import Wallet

def test_basic_operations():
    """Test basic wallet operations - creation, deposits, and balances."""
    logger.info("Testing basic wallet operations")
    
    # Create wallet with initial USDT balance
    wallet = Wallet(initial_balance=10000.0, base_currency="USDT", name="Test Wallet")
    
    # Check initial balance
    initial_balance = wallet.get_balance("USDT")
    logger.info(f"Initial balance: {initial_balance} USDT")
    assert initial_balance == 10000.0, f"Expected 10000.0 USDT, got {initial_balance}"
    
    # Add deposit
    wallet.add_deposit("BTC", 0.5, source="external")
    btc_balance = wallet.get_balance("BTC")
    logger.info(f"After deposit: {btc_balance} BTC")
    assert btc_balance == 0.5, f"Expected 0.5 BTC, got {btc_balance}"
    
    # Show all balances
    balances = wallet.get_all_balances()
    logger.info(f"All balances: {balances}")
    
    return wallet

def test_trading_operations(wallet):
    """Test trading operations - buying and selling assets."""
    logger.info("Testing trading operations")
    
    # Simulate buying BTC with USDT
    btc_price = 50000.0
    usdt_amount = 5000.0
    btc_amount = usdt_amount / btc_price
    
    # Record buy trade
    trade = wallet.add_trade(
        trade_type="buy",
        from_currency="USDT",
        to_currency="BTC",
        from_amount=usdt_amount,
        to_amount=btc_amount,
        price=btc_price,
        fee=10.0,
        fee_currency="USDT",
        exchange="binance",
        external_id="test-trade-1"
    )
    
    logger.info(f"Executed buy trade: {btc_amount} BTC at {btc_price} USDT")
    
    # Check updated balances
    usdt_balance = wallet.get_balance("USDT")
    btc_balance = wallet.get_balance("BTC")
    logger.info(f"After buy: {usdt_balance} USDT, {btc_balance} BTC")
    
    # Expected: initial 10000 - 5000 - 10 fee = 4990 USDT
    assert abs(usdt_balance - 4990.0) < 0.01, f"Expected ~4990 USDT, got {usdt_balance}"
    # Expected: 0.5 + 5000/50000 = 0.5 + 0.1 = 0.6 BTC
    assert abs(btc_balance - 0.6) < 0.01, f"Expected ~0.6 BTC, got {btc_balance}"
    
    # Simulate BTC price increase
    new_btc_price = 55000.0
    
    # Sell part of BTC
    btc_to_sell = 0.2
    usdt_to_receive = btc_to_sell * new_btc_price
    
    # Record sell trade
    trade = wallet.add_trade(
        trade_type="sell",
        from_currency="BTC",
        to_currency="USDT",
        from_amount=btc_to_sell,
        to_amount=usdt_to_receive,
        price=new_btc_price,
        fee=11.0,
        fee_currency="USDT",
        exchange="binance",
        external_id="test-trade-2"
    )
    
    logger.info(f"Executed sell trade: {btc_to_sell} BTC at {new_btc_price} USDT")
    
    # Check updated balances
    usdt_balance = wallet.get_balance("USDT")
    btc_balance = wallet.get_balance("BTC")
    logger.info(f"After sell: {usdt_balance} USDT, {btc_balance} BTC")
    
    # Expected: 4990 + (0.2 * 55000) - 11 fee = 4990 + 11000 - 11 = 15979 USDT
    assert abs(usdt_balance - 15979.0) < 0.01, f"Expected ~15979 USDT, got {usdt_balance}"
    # Expected: 0.6 - 0.2 = 0.4 BTC
    assert abs(btc_balance - 0.4) < 0.01, f"Expected 0.4 BTC, got {btc_balance}"
    
    return wallet

def test_portfolio_valuation(wallet):
    """Test portfolio valuation with current market prices."""
    logger.info("Testing portfolio valuation")
    
    # Define current market prices
    current_prices = {
        "BTC": 58000.0,
        "ETH": 3500.0
    }
    
    # Deposit some ETH for diversification
    wallet.add_deposit("ETH", 2.0, source="external")
    
    # Update wallet with current prices and calculate total value
    valuation = wallet.calculate_total_value(current_prices)
    
    # Log the valuation details
    logger.info(f"Portfolio valuation: {valuation['total_value']} {wallet.base_currency}")
    logger.info(f"Base currency balance: {valuation['base_balance']} {wallet.base_currency}")
    
    for currency, details in valuation["holdings"].items():
        logger.info(f"{currency}: {details['balance']} @ {details['price']} = {details['value']} {wallet.base_currency}")
    
    # Check performance metrics
    metrics = wallet.get_performance_metrics()
    logger.info(f"Performance metrics: {metrics}")
    
    # Calculate expected total value
    expected_usdt = wallet.get_balance("USDT")
    expected_btc_value = wallet.get_balance("BTC") * current_prices["BTC"]
    expected_eth_value = wallet.get_balance("ETH") * current_prices["ETH"]
    expected_total = expected_usdt + expected_btc_value + expected_eth_value
    
    assert abs(valuation["total_value"] - expected_total) < 0.01, \
        f"Expected valuation ~{expected_total}, got {valuation['total_value']}"
    
    return wallet

def test_trade_history(wallet):
    """Test trade history retrieval and filtering."""
    logger.info("Testing trade history functionality")
    
    # Get all trade history
    all_trades = wallet.get_trade_history()
    logger.info(f"Found {len(all_trades)} trades in history")
    
    # Filter by trade type
    buy_trades = wallet.get_trade_history(trade_type="buy")
    sell_trades = wallet.get_trade_history(trade_type="sell")
    logger.info(f"Buy trades: {len(buy_trades)}, Sell trades: {len(sell_trades)}")
    
    # Filter by currency
    btc_trades = wallet.get_trade_history(currency="BTC")
    eth_trades = wallet.get_trade_history(currency="ETH")
    logger.info(f"BTC trades: {len(btc_trades)}, ETH trades: {len(eth_trades)}")
    
    return wallet

def test_wallet_persistence(wallet):
    """Test saving and loading wallet state."""
    logger.info("Testing wallet persistence")
    
    # Ensure the db directory exists
    os.makedirs("db", exist_ok=True)
    
    # Save wallet to file
    file_path = "db/test_wallet.json"
    success = wallet.save_to_file(file_path)
    assert success, "Failed to save wallet to file"
    logger.info(f"Wallet saved to {file_path}")
    
    # Load wallet from file
    loaded_wallet = Wallet.load_from_file(file_path)
    logger.info(f"Loaded wallet: {loaded_wallet.name}, ID: {loaded_wallet.wallet_id}")
    
    # Verify balances match
    original_balances = wallet.get_all_balances()
    loaded_balances = loaded_wallet.get_all_balances()
    
    for currency, amount in original_balances.items():
        assert currency in loaded_balances, f"Currency {currency} missing in loaded wallet"
        assert abs(loaded_balances[currency] - amount) < 0.01, \
            f"Balance mismatch for {currency}: expected {amount}, got {loaded_balances[currency]}"
    
    logger.info("Wallet persistence test passed - balances match")
    
    return loaded_wallet

def run_all_tests():
    """Run all wallet tests in sequence."""
    try:
        wallet = test_basic_operations()
        wallet = test_trading_operations(wallet)
        wallet = test_portfolio_valuation(wallet)
        wallet = test_trade_history(wallet)
        wallet = test_wallet_persistence(wallet)
        
        logger.info("All wallet tests completed successfully!")
        return True
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 