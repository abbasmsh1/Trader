# AI Crypto Trading Dashboard

A sophisticated dashboard for AI-powered cryptocurrency trading, featuring multiple trading agents with distinct personalities competing to turn $20 into $100.

![AI Crypto Trading Dashboard](https://i.imgur.com/placeholder.png)

## Features

### Multiple AI Trading Agents

The system features three distinct AI trading agents, each with a unique personality and trading style:

- **Warren Buffett AI**: A value investor focused on long-term growth and intrinsic value. Conservative approach with emphasis on capital preservation.
- **Elon Musk AI**: A tech disruptor with high risk tolerance, focusing on innovation and momentum. Particularly interested in meme coins.
- **Technical Trader**: A systematic, data-driven trader following technical indicators and chart patterns. Emotionless and disciplined.

### Comprehensive Market Data

- Real-time price data for multiple cryptocurrencies
- Support for both USDT pairs (BTC/USDT, ETH/USDT, etc.) and coin-to-coin pairs (BTC/ETH, ETH/BTC, etc.)
- Includes major coins, alt coins, and meme coins
- Technical indicators: SMA, RSI, MACD, Bollinger Bands

### Advanced Trading Features

- Automatic trading based on AI signals
- Memory system to continue trading sessions where you left off
- Detailed portfolio tracking with USDT value calculations
- Goal tracking ($20 to $100 challenge)
- Trade history and performance metrics

### Modern UI

- Dark-themed, responsive dashboard
- Interactive charts with multiple layout options
- Detailed holdings breakdown with percentage visualizations
- Real-time signal generation and trade execution
- Wallet composition visualization

## Getting Started

### Prerequisites

- Python 3.7+
- Pip package manager

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-crypto-trading.git
   cd ai-crypto-trading
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   Create a `.env` file in the project root with your exchange API keys:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   ```

### Running the Dashboard

Start the dashboard with:
```
python main.py
```

The dashboard will be available at `http://127.0.0.1:8050/`

## Usage Guide

### Trading View

The main trading view displays charts for multiple cryptocurrencies. You can:

- View price charts for all supported coins
- Select different timeframes (1h, 4h, 1d, etc.)
- Choose technical indicators to display
- Switch between candlestick, line, and OHLC chart styles
- Adjust the chart layout (2x2, 3x3, 4x4 grid)
- Enable/disable auto-trading

### Traders Comparison

The traders comparison view allows you to:

- Compare performance of different AI trading agents
- View detailed portfolio breakdowns
- See the USDT value of all crypto holdings
- Track progress toward the $100 goal
- Review trade history and execution details

## How It Works

1. **Market Analysis**: The system continuously fetches market data for all configured cryptocurrency pairs.
2. **AI Signal Generation**: Each AI agent analyzes the market data based on their unique personality and strategy.
3. **Trade Execution**: If auto-trading is enabled, trades are executed based on the generated signals.
4. **Portfolio Management**: The system tracks all holdings and calculates performance metrics.
5. **Memory System**: Trading state is saved periodically, allowing the system to continue from where it left off.

## Project Structure

- `main.py`: Main application file with dashboard and trading system
- `data/`: Data handling modules
  - `market_data.py`: Market data fetching and processing
  - `wallet.py`: Wallet management and performance tracking
- `agents/`: Trading agent implementations
  - `base_agent.py`: Base class for all trading agents
  - `value_investor.py`: Warren Buffett AI implementation
  - `tech_disruptor.py`: Elon Musk AI implementation
  - `trend_follower.py`: Technical Trader implementation
- `config/`: Configuration files
  - `settings.py`: System parameters and default settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor. 