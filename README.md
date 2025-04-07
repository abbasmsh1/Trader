# Crypto Trader

An agent-based cryptocurrency trading system that uses a hierarchical architecture to make trading decisions and manage a portfolio of digital assets.

## Features

- **Agent-Based Architecture**: Hierarchical system with specialized agents for different tasks.
- **Portfolio Management**: Intelligent asset allocation and risk management.
- **Multiple Strategies**: Support for various trading strategies including trend following and mean reversion.
- **Backtesting**: Test strategies against historical data before live trading.
- **Wallet Management**: Track deposits, withdrawals, and trades in multiple currencies.
- **Web Dashboard**: Monitor performance, portfolio status, and control the system.
- **Demo Mode**: Practice trading without risking real money.

## Structure

The project is organized into the following components:

- `agents/`: Trading agents and hierarchical agent structure
- `models/`: Core data models and classes (Wallet, Order, Position, etc.)
- `strategies/`: Trading strategy implementations
- `data/`: Data management and market data providers
- `utils/`: Utility functions and helper classes
- `config/`: Configuration files
- `frontend/`: Web dashboard and user interface

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-trader.git
   cd crypto-trader
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install TA-Lib (may require additional steps depending on your OS):
   - On Ubuntu: `apt-get install ta-lib`
   - On macOS: `brew install ta-lib`
   - On Windows: Download and install from [TA-Lib Binary](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

5. Configure your API keys and settings in `config/config.json`

## Usage

### Running the Trading System

To start the trading system:

```
python main.py
```

### Running the Web Dashboard

To start the web dashboard:

```
cd frontend
python app.py
```

Then open your browser to `http://localhost:5000`

### Demo Mode

If no exchange API keys are configured, the system will run in demo mode, simulating trades and portfolio management.

## Agent Architecture

The system uses a hierarchical agent architecture:

1. **System Controller**: The top-level agent that coordinates all activities
2. **Portfolio Manager**: Manages asset allocation and risk
3. **Strategy Agents**: Implements specific trading strategies
4. **Execution Agent**: Handles order execution and exchange interaction
5. **Data Agents**: Collects and processes market data

## Development

### Adding a New Strategy

To add a new trading strategy:

1. Create a new file in `strategies/` directory
2. Implement the strategy class inheriting from `BaseStrategy`
3. Register your strategy in `config/config.json`

### Adding Support for a New Exchange

The system uses CCXT library for exchange integration. To add a new exchange:

1. Ensure the exchange is supported by CCXT
2. Add API configuration in `config/config.json`
3. Test connectivity with the exchange

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves significant risk. Use at your own risk. 