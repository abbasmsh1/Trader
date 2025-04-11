# Multi-Agent AI Trading System with Famous Trader Personas

A cryptocurrency trading system featuring AI agents modeled after famous real-world traders (Warren Buffett, George Soros, Jim Simons, Peter Lynch). Each agent has a unique personality, trading strategy, and risk profile. The system uses Deep Q-Networks with continuous reinforcement learning for decision making.

## Features

- **Personalized AI Trader Agents**: Each agent mimics the philosophy and decision style of a famous trader
- **Multiple Cryptocurrencies**: Trade BTC, ETH, SOL, and more against USDT
- **Real-time & Historical Data**: Uses both real-time price feeds and historical data
- **Simulated Trading**: Test strategies with simulated wallets (Binance Testnet support planned)
- **State Persistence**: Save and restore the entire system state
- **Minimum Trade Limit**: $5 minimum trade value enforced by the system
- **Interactive Dashboard**: Monitor trader performance, portfolio values, and transactions

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-trader.git
   cd crypto-trader
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating a `.env` file:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   ```

## Usage

### Running in Demo Mode

Demo mode uses simulated market data and does not require API keys:

```bash
python main.py --demo
```

### Running with Testnet

For testing with real APIs but simulated money:

```bash
python main.py --testnet
```

### Running the Web Interface

The web interface provides a dashboard to monitor and control the trading system:

```bash
cd frontend
python app.py
```

Then open your browser to `http://localhost:5000`

## Trader Agent Personalities

### Buffett Trader (Value Investing)
- **Strategy**: Long-term value investing
- **Focus**: Fundamental analysis, margin of safety
- **Holding Period**: Long (weeks to months)
- **Risk Profile**: Low risk, steady returns

### Soros Trader (Macro Trading)
- **Strategy**: Reflexivity and market feedback loops
- **Focus**: Macro trends, regime shifts, sentiment
- **Holding Period**: Medium (days to weeks)
- **Risk Profile**: Medium-high risk, high potential returns

### Simons Trader (Quantitative)
- **Strategy**: Statistical arbitrage and mean reversion
- **Focus**: Pattern detection, correlations, statistical indicators
- **Holding Period**: Short (minutes to days)
- **Risk Profile**: Medium risk, consistent returns

### Lynch Trader (Growth Investing)
- **Strategy**: Growth at a reasonable price
- **Focus**: Emerging trends, momentum, "tenbagger" potential
- **Holding Period**: Medium-long (weeks to months)
- **Risk Profile**: Medium risk, high growth potential

## Project Structure

```
/crypto_trader/
│
├── agents/               # Agent implementations
│   ├── trader/           # Trading agent implementations
│   │   ├── buffett_trader.py
│   │   ├── soros_trader.py
│   │   ├── simons_trader.py
│   │   ├── lynch_trader.py
│   │   └── base_trader.py
│   ├── system_controller.py
│   └── base_agent.py
│
├── config/               # Configuration files
│   └── config.json
│
├── data/                 # Data storage
│   └── historical/       # Historical market data
│
├── db/                   # Database implementations
│   └── db_handler.py
│
├── frontend/             # Web interface
│   ├── static/
│   ├── templates/
│   └── app.py
│
├── models/               # Model implementations
│   └── wallet.py
│
├── utils/                # Utility functions
│   └── agent_manager.py
│
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inspired by the trading strategies of Warren Buffett, George Soros, Jim Simons, and Peter Lynch
- Uses CCXT for cryptocurrency exchange connectivity
- Built with PyTorch for reinforcement learning 