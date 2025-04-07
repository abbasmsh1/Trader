.PHONY: setup run run-once backtest optimize status clean clean-logs clean-db help wallet-demo wallet-test

help:
	@echo "Crypto Trader System Makefile"
	@echo "-----------------------------"
	@echo "make setup     - Create necessary directories and setup environment"
	@echo "make run       - Run the trading system in continuous mode"
	@echo "make run-once  - Run a single cycle of the trading system"
	@echo "make backtest  - Run backtest mode (not implemented yet)"
	@echo "make optimize  - Run optimization mode (not implemented yet)"
	@echo "make status    - Check trading system status"
	@echo "make wallet-demo - Run wallet demonstration script"
	@echo "make wallet-test - Run wallet tests"
	@echo "make clean     - Clean temporary files, logs, and database"
	@echo "make clean-logs - Clean only log files"
	@echo "make clean-db   - Clean only database files"

setup:
	@echo "Setting up Crypto Trader system..."
	mkdir -p data db logs
	[ -f .env ] || cp .env.example .env
	@echo "Setup complete. Please edit .env file with your API keys and configuration."

run:
	@echo "Running Crypto Trader system in continuous mode..."
	python run_trader.py --mode run

run-once:
	@echo "Running a single cycle of the Crypto Trader system..."
	python run_trader.py --mode run --once

backtest:
	@echo "Running backtest mode..."
	python run_trader.py --mode backtest

optimize:
	@echo "Running optimization mode..."
	python run_trader.py --mode optimize

status:
	@echo "Checking Crypto Trader system status..."
	python run_trader.py --mode status

wallet-demo:
	@echo "Running wallet demonstration script..."
	python run_wallet_demo.py

wallet-test:
	@echo "Running wallet tests..."
	python tests/test_wallet.py

clean-logs:
	@echo "Cleaning log files..."
	rm -f *.log logs/*.log

clean-db:
	@echo "Cleaning database files..."
	rm -f db/*.db

clean: clean-logs clean-db
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} + 