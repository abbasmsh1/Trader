# AI Crypto Trader Simulation

A system of AI agents modeled after famous traders, each with a unique personality and trading style. The agents trade cryptocurrencies using real-time Binance chart data, starting with virtual wallets of $20, aiming to grow their funds to $100 as quickly as possible.

## Features

- 5 AI trader agents with distinct personalities and trading styles
- Real-time cryptocurrency trading using Binance API data
- Interactive dashboard showing market charts, trader portfolios, and agent interactions
- Agent communication system where traders discuss strategies and attempt to persuade each other

## Trader Agents

1. **Warren Weber** - Cautious value trader; buys undervalued coins and holds long-term
2. **Sonia Soros** - Aggressive macro trader; bets big on market-moving news or breakouts
3. **Pete the Scalper** - Hyperactive day trader; makes rapid small trades based on 1-minute charts
4. **Johnny Bollinger** - Technical analyst; trades based on Bollinger Bands and RSI signals
5. **Lena Lynch** - Growth trader; focuses on coins with strong fundamentals and steady uptrends

## Project Structure

- `backend/` - Python backend for agent logic and Binance API integration
- `frontend/` - React.js frontend for the dashboard
- `docs/` - Documentation and design files

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- Binance API key (for real-time data)

### Installation

1. Clone the repository
2. Install backend dependencies: `pip install -r requirements.txt`
3. Install frontend dependencies: `cd frontend && npm install`
4. Configure your Binance API key in `config.py`
5. Start the backend: `python backend/app.py`
6. Start the frontend: `cd frontend && npm start`

## License

MIT 