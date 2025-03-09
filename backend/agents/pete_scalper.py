import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class PeteScalper(Trader):
    """
    Pete the Scalper - Hyperactive day trader; makes rapid small trades based on 1-minute charts.
    """
    
    def __init__(self):
        """Initialize the Pete the Scalper trader agent."""
        super().__init__(
            name="Pete the Scalper",
            personality="Hyperactive, impatient, quick-thinking, and detail-oriented",
            trading_style="Scalping, high-frequency trading, short-term price action",
            backstory="A former high-frequency trading algorithm programmer who now applies his lightning-fast reflexes to crypto markets. Pete thrives on volatility and makes dozens of small trades daily, aiming to capture tiny price movements for consistent profits. He rarely holds positions for more than a few minutes and is constantly glued to the 1-minute charts."
        )
        
        # Trading parameters
        self.min_price_move = 0.5  # Minimum price move to consider (0.5%)
        self.take_profit = 1.5  # Take profit at 1.5%
        self.stop_loss = -1.0  # Cut losses at 1%
        self.max_position_size = 0.3  # Maximum 30% of portfolio in a single asset
        self.max_holding_time = timedelta(minutes=15)  # Maximum time to hold a position
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.analyzed_pairs = {}  # Symbol -> analysis result
        self.last_analysis_time = datetime.now() - timedelta(minutes=1)  # Force initial analysis
        self.trade_count = 0  # Number of trades made
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the market data and make trading decisions.
        
        Args:
            market_data: The current market data
            
        Returns:
            A dictionary containing the analysis results and trading decisions
        """
        current_time = datetime.now()
        result = {
            'analysis': {},
            'trades': []
        }
        
        # Pete analyzes the market very frequently (every minute)
        if current_time - self.last_analysis_time < timedelta(minutes=1):
            return result
        
        self.last_analysis_time = current_time
        logger.info(f"{self.name} is analyzing the market...")
        
        # Get current prices and portfolio value
        current_prices = {
            symbol: data['price'] 
            for symbol, data in market_data.items()
        }
        portfolio_value = self.get_portfolio_value(current_prices)
        
        # Analyze each trading pair
        for symbol, data in market_data.items():
            # Skip if not enough data
            if 'indicators' not in data:
                continue
            
            # Extract data
            price = data['price']
            
            # Get the base symbol (e.g., 'BTC' from 'BTCUSDT')
            base_symbol = symbol.replace('USDT', '')
            
            # Check if we already own this cryptocurrency
            holding_amount = self.wallet.get(base_symbol, 0)
            holding_value = holding_amount * price
            
            # Calculate position size as percentage of portfolio
            position_size = holding_value / portfolio_value if portfolio_value > 0 else 0
            
            # Initialize analysis
            analysis = {
                'symbol': symbol,
                'price': price,
                'holding_amount': holding_amount,
                'holding_value': holding_value,
                'position_size': position_size,
                'short_term_signal': 'neutral',  # Default signal
                'action': 'hold'  # Default action
            }
            
            # Check for short-term signals in 1-minute timeframe
            if 'indicators' in data and '1m' in data['indicators']:
                indicators = data['indicators']['1m']
                
                # Check RSI for overbought/oversold
                if 'RSI' in indicators:
                    rsi = indicators['RSI']
                    analysis['rsi'] = rsi['value']
                    
                    if rsi['value'] < 30:
                        analysis['short_term_signal'] = 'buy'
                    elif rsi['value'] > 70:
                        analysis['short_term_signal'] = 'sell'
                
                # Check MACD for bullish/bearish signals
                if 'MACD' in indicators:
                    macd = indicators['MACD']
                    
                    if macd['interpretation'] == 'bullish_crossover':
                        analysis['short_term_signal'] = 'buy'
                    elif macd['interpretation'] == 'bearish_crossover':
                        analysis['short_term_signal'] = 'sell'
                
                # Check Bollinger Bands for price action
                if 'BOLLINGER' in indicators:
                    bollinger = indicators['BOLLINGER']
                    
                    if bollinger['interpretation'] == 'oversold':
                        analysis['short_term_signal'] = 'buy'
                    elif bollinger['interpretation'] == 'overbought':
                        analysis['short_term_signal'] = 'sell'
            
            # Check recent price action using trades data
            if 'trades' in data and len(data['trades']) >= 2:
                recent_trades = data['trades'][:5]  # Last 5 trades
                
                # Calculate average price change
                price_changes = []
                for i in range(1, len(recent_trades)):
                    prev_price = float(recent_trades[i-1]['price'])
                    curr_price = float(recent_trades[i]['price'])
                    change_pct = ((curr_price - prev_price) / prev_price) * 100
                    price_changes.append(change_pct)
                
                if price_changes:
                    avg_change = sum(price_changes) / len(price_changes)
                    analysis['recent_price_change'] = avg_change
                    
                    # Strong short-term momentum
                    if avg_change > self.min_price_move:
                        analysis['short_term_signal'] = 'buy'
                    elif avg_change < -self.min_price_move:
                        analysis['short_term_signal'] = 'sell'
            
            # Determine if we should buy
            if analysis['short_term_signal'] == 'buy' and position_size < self.max_position_size:
                # Calculate how much to buy
                available_usdt = self.wallet.get('USDT', 0)
                
                # Limit position size
                max_additional_position = self.max_position_size - position_size
                max_buy_amount = min(available_usdt, portfolio_value * max_additional_position)
                
                # Only buy if we have enough USDT (at least $1)
                if max_buy_amount >= 1.0:
                    # Pete makes small trades
                    buy_amount = min(max_buy_amount * 0.3, 5.0)  # Max $5 per trade
                    
                    analysis['action'] = 'buy'
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': buy_amount,
                        'reason': f"Short-term buy signal ({analysis['short_term_signal']})"
                    })
                    
                    # Record purchase price and time
                    self.purchase_prices[base_symbol] = price
                    self.purchase_times[base_symbol] = current_time
                    self.trade_count += 1
            
            # Determine if we should sell
            elif holding_amount > 0:
                purchase_price = self.purchase_prices.get(base_symbol)
                purchase_time = self.purchase_times.get(base_symbol)
                
                if purchase_price and purchase_time:
                    price_change_from_purchase = ((price - purchase_price) / purchase_price) * 100
                    holding_duration = current_time - purchase_time
                    
                    # Take profit, cut losses, or exit if held too long
                    if price_change_from_purchase >= self.take_profit:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Take profit (+{price_change_from_purchase:.2f}%)"
                        })
                        self.trade_count += 1
                    elif price_change_from_purchase <= self.stop_loss:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Stop loss ({price_change_from_purchase:.2f}%)"
                        })
                        self.trade_count += 1
                    elif holding_duration >= self.max_holding_time:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Max holding time reached ({holding_duration.total_seconds() / 60:.1f} minutes)"
                        })
                        self.trade_count += 1
                    elif analysis['short_term_signal'] == 'sell':
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Short-term sell signal ({analysis['short_term_signal']})"
                        })
                        self.trade_count += 1
            
            # Store the analysis
            result['analysis'][symbol] = analysis
            self.analyzed_pairs[symbol] = analysis
        
        return result
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List[Trader]) -> Optional[str]:
        """
        Generate a message to share with other traders.
        
        Args:
            market_data: The current market data
            other_traders: List of other traders
            
        Returns:
            A message string, or None if no message is generated
        """
        # Pete is very chatty (50% chance)
        if random.random() > 0.5:
            return None
        
        # Find a cryptocurrency with a strong short-term signal
        strong_signals = []
        
        for symbol, analysis in self.analyzed_pairs.items():
            if 'short_term_signal' in analysis and analysis['short_term_signal'] in ['buy', 'sell']:
                strong_signals.append((symbol, analysis['short_term_signal']))
        
        if strong_signals:
            symbol, signal = random.choice(strong_signals)
            
            if signal == 'buy':
                messages = [
                    f"Quick buy opportunity on {symbol}! The 1-minute chart is showing a perfect entry point right now.",
                    f"Just scalped a quick position in {symbol}. In and out for a quick 1% - that's how it's done!",
                    f"The {symbol} order book is stacked with buy orders. I'm jumping in for a quick flip.",
                    f"RSI just hit oversold on the {symbol} 1-minute chart. Classic scalping setup!",
                    f"Catching this {symbol} bounce! Already up 0.5% in 30 seconds. This is what scalping is all about!"
                ]
            else:  # sell
                messages = [
                    f"Dumping my {symbol} position NOW. The 1-minute chart is rolling over.",
                    f"Just took profits on {symbol} - small gains add up! That's scalp #{self.trade_count} today.",
                    f"The {symbol} momentum is fading fast. I'm out with a quick profit.",
                    f"RSI just hit overbought on the {symbol} 1-minute chart. Selling before the pullback!",
                    f"Selling {symbol} into strength. Don't be greedy - take those small profits!"
                ]
            
            return random.choice(messages)
        
        # General messages about scalping
        general_messages = [
            f"Made {self.trade_count} trades so far today. Small profits, high frequency - that's the scalper's way!",
            "The 1-minute charts are where the real action is. Who has time for daily charts?",
            "Just watching for quick reversals. In and out - that's how I like my trades.",
            "Scalping is all about volume and volatility. The more trades, the more opportunities!",
            "Who needs long-term investing when you can make 20 small profits a day?"
        ]
        return random.choice(general_messages)
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """
        Respond to a message from another trader.
        
        Args:
            message: The message to respond to
            market_data: The current market data
            
        Returns:
            A response message string, or None if no response is generated
        """
        # Pete responds quickly (70% chance)
        if random.random() > 0.7:
            return None
        
        content = message['content'].lower()
        sender_name = message['trader_name']
        
        # Respond to messages about long-term investing
        if any(term in content for term in ['long-term', 'hold', 'hodl', 'patient', 'value']):
            responses = [
                f"{sender_name}, why wait months for 20% when you can make 1% twenty times in a week? Scalping is the way!",
                f"Long-term? No thanks, {sender_name}. I've already made 3 profitable trades while you were typing that message!",
                f"With all due respect, {sender_name}, HODLing is for people who don't know how to read 1-minute charts.",
                f"That's cute, {sender_name}, but I prefer to make money TODAY, not someday in the distant future."
            ]
            return random.choice(responses)
        
        # Respond to messages about specific cryptocurrencies
        for symbol in TRADING_PAIRS:
            if symbol in content:
                # Check if we've analyzed this pair
                if symbol in self.analyzed_pairs:
                    analysis = self.analyzed_pairs[symbol]
                    
                    if 'short_term_signal' in analysis:
                        if analysis['short_term_signal'] == 'buy':
                            responses = [
                                f"{symbol} is showing a nice scalping setup right now, {sender_name}. The 1-minute RSI just bounced off oversold.",
                                f"I'm watching {symbol} too, {sender_name}! Just waiting for a quick pullback to scalp a long position.",
                                f"The bid-ask spread on {symbol} is tight right now, {sender_name}. Perfect for a quick in-and-out trade."
                            ]
                            return random.choice(responses)
                        elif analysis['short_term_signal'] == 'sell':
                            responses = [
                                f"I'm actually looking to short {symbol} on the next bounce, {sender_name}. The 1-minute momentum is clearly down.",
                                f"Just took profits on my {symbol} position, {sender_name}. Never hold too long - that's the scalper's rule!",
                                f"The selling pressure on {symbol} is increasing, {sender_name}. I'd wait for a better entry if I were you."
                            ]
                            return random.choice(responses)
                
                # General response about the cryptocurrency
                responses = [
                    f"I've scalped {symbol} 3 times already today, {sender_name}. The volatility is perfect for quick trades.",
                    f"The 1-minute chart on {symbol} is what you should be watching, {sender_name}. Forget about the daily.",
                    f"I could show you how to make 10 quick trades on {symbol} today, {sender_name}. Much better than waiting for some big move."
                ]
                return random.choice(responses)
        
        # General responses
        general_responses = [
            f"While you're planning your next move, {sender_name}, I've already made three trades. Speed is everything in this market!",
            f"Interesting perspective, {sender_name}, but I prefer to focus on what's happening in the next 5 minutes, not the next 5 months.",
            f"That's one approach, {sender_name}, but have you tried scalping? Much more exciting and profitable if you ask me!",
            f"I respect your strategy, {sender_name}, but I've found that quick in-and-out trades add up to much bigger profits over time."
        ]
        return random.choice(general_responses) 