import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class WarrenWeber(Trader):
    """
    Warren Weber - Cautious value trader; buys undervalued coins and holds long-term.
    """
    
    def __init__(self):
        """Initialize the Warren Weber trader agent."""
        super().__init__(
            name="Warren Weber",
            personality="Cautious, patient, methodical, and risk-averse",
            trading_style="Value investing, long-term holding, fundamental analysis",
            backstory="A legendary value investor who believes in buying quality assets at a discount and holding them for the long term. Warren has a knack for identifying undervalued cryptocurrencies with strong fundamentals and patiently waiting for the market to recognize their true value."
        )
        
        # Trading parameters
        self.min_holding_period = timedelta(minutes=30)  # Minimum time to hold before selling
        self.buy_threshold = -5.0  # Buy when price drops by 5% or more in 24h
        self.sell_threshold = 15.0  # Sell when price increases by 15% or more from purchase
        self.max_position_size = 0.4  # Maximum 40% of portfolio in a single asset
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.analyzed_pairs = {}  # Symbol -> analysis result
        self.last_analysis_time = datetime.now() - timedelta(minutes=10)  # Force initial analysis
    
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
        
        # Only perform deep analysis every 10 minutes
        if current_time - self.last_analysis_time < timedelta(minutes=10):
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
            if 'price_change_percent_24h' not in data:
                continue
            
            # Extract data
            price = data['price']
            price_change_24h = data['price_change_percent_24h']
            
            # Get the base symbol (e.g., 'BTC' from 'BTCUSDT')
            base_symbol = symbol.replace('USDT', '')
            
            # Check if we already own this cryptocurrency
            holding_amount = self.wallet.get(base_symbol, 0)
            holding_value = holding_amount * price
            
            # Calculate position size as percentage of portfolio
            position_size = holding_value / portfolio_value if portfolio_value > 0 else 0
            
            # Analyze the cryptocurrency
            analysis = {
                'symbol': symbol,
                'price': price,
                'price_change_24h': price_change_24h,
                'holding_amount': holding_amount,
                'holding_value': holding_value,
                'position_size': position_size,
                'is_undervalued': price_change_24h <= self.buy_threshold,
                'is_overvalued': False,  # Warren doesn't believe in "overvalued" - only sells on profit target
                'action': 'hold'  # Default action
            }
            
            # Determine if we should buy
            if analysis['is_undervalued'] and position_size < self.max_position_size:
                # Calculate how much to buy
                available_usdt = self.wallet.get('USDT', 0)
                
                # Limit position size
                max_additional_position = self.max_position_size - position_size
                max_buy_amount = min(available_usdt, portfolio_value * max_additional_position)
                
                # Only buy if we have enough USDT (at least $1)
                if max_buy_amount >= 1.0:
                    # Warren is cautious, so he only uses a portion of available funds
                    buy_amount = max_buy_amount * 0.5
                    
                    analysis['action'] = 'buy'
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': buy_amount,
                        'reason': f"Undervalued (-{abs(price_change_24h):.2f}% in 24h)"
                    })
                    
                    # Record purchase price and time
                    self.purchase_prices[base_symbol] = price
                    self.purchase_times[base_symbol] = current_time
            
            # Determine if we should sell
            elif holding_amount > 0:
                # Check if we've held long enough
                purchase_time = self.purchase_times.get(base_symbol)
                purchase_price = self.purchase_prices.get(base_symbol)
                
                if purchase_time and purchase_price:
                    holding_duration = current_time - purchase_time
                    price_change_from_purchase = ((price - purchase_price) / purchase_price) * 100
                    
                    # Sell if we've reached our profit target and held for minimum period
                    if price_change_from_purchase >= self.sell_threshold and holding_duration >= self.min_holding_period:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Profit target reached (+{price_change_from_purchase:.2f}%)"
                        })
            
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
        # Warren only speaks when he has something valuable to say (20% chance)
        if random.random() > 0.2:
            return None
        
        # Find the most undervalued cryptocurrency
        most_undervalued = None
        largest_drop = 0
        
        for symbol, data in market_data.items():
            if 'price_change_percent_24h' in data and data['price_change_percent_24h'] < 0:
                drop = abs(data['price_change_percent_24h'])
                if drop > largest_drop:
                    largest_drop = drop
                    most_undervalued = symbol
        
        if most_undervalued:
            messages = [
                f"I've been watching {most_undervalued} closely. It's down {largest_drop:.2f}% in the last 24 hours, which presents a potential value opportunity. Remember, be greedy when others are fearful.",
                f"The market seems to be undervaluing {most_undervalued}. This kind of short-term volatility is exactly what creates long-term opportunities for patient investors.",
                f"I'm considering adding to my {most_undervalued} position. The fundamentals haven't changed, but the price has dropped significantly. That's the essence of value investing.",
                f"When you buy quality assets at a discount, time is your friend. {most_undervalued} looks like it's on sale today.",
                f"The secret to successful investing is to focus on the intrinsic value, not the daily price movements. That said, {most_undervalued}'s recent drop makes it even more attractive."
            ]
            return random.choice(messages)
        
        # General wisdom if no undervalued crypto found
        general_messages = [
            "The market is quite volatile today. Remember that in the short run, the market is a voting machine, but in the long run, it's a weighing machine.",
            "I prefer to focus on the long-term value rather than short-term price fluctuations. Patience is the key to successful investing.",
            "The best investments are often the ones that feel uncomfortable to make. Buy when others are fearful, sell when others are greedy.",
            "I'm holding my positions for now. Quality assets tend to appreciate over time if you have the patience to wait.",
            "The most important quality for an investor is temperament, not intellect. Don't let market volatility shake your conviction in sound investments."
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
        # Warren only responds occasionally (30% chance)
        if random.random() > 0.3:
            return None
        
        content = message['content'].lower()
        sender_name = message['trader_name']
        
        # Respond to messages about quick profits or day trading
        if any(term in content for term in ['quick', 'fast', 'day trade', 'scalp', 'short-term']):
            responses = [
                f"{sender_name}, I respectfully disagree with your approach. The stock market is designed to transfer money from the active to the patient.",
                f"I understand your enthusiasm, {sender_name}, but remember that the market is a device for transferring money from the impatient to the patient.",
                f"While I appreciate your perspective, {sender_name}, I've found that trying to time the market is far less profitable than time in the market.",
                f"{sender_name}, in my experience, the only value of stock forecasters is to make fortune tellers look good."
            ]
            return random.choice(responses)
        
        # Respond to messages about specific cryptocurrencies
        for symbol in TRADING_PAIRS:
            if symbol in content:
                base_symbol = symbol.replace('USDT', '')
                
                # Check if we've analyzed this pair
                if symbol in self.analyzed_pairs:
                    analysis = self.analyzed_pairs[symbol]
                    
                    if analysis['is_undervalued']:
                        responses = [
                            f"I agree that {symbol} looks interesting, {sender_name}. It appears to be trading below its intrinsic value right now.",
                            f"{sender_name}, I've been watching {symbol} as well. The recent price drop presents a potential opportunity for patient investors.",
                            f"You make a good point about {symbol}, {sender_name}. I believe in buying quality assets when they're temporarily undervalued."
                        ]
                        return random.choice(responses)
                
                # General response about the cryptocurrency
                responses = [
                    f"I'm taking a wait-and-see approach with {symbol}, {sender_name}. I prefer to thoroughly analyze fundamentals before making investment decisions.",
                    f"Interesting thoughts on {symbol}, {sender_name}. I tend to focus more on long-term value than short-term price movements.",
                    f"Thank you for sharing your perspective on {symbol}, {sender_name}. I'll consider it as part of my broader analysis."
                ]
                return random.choice(responses)
        
        # General responses
        general_responses = [
            f"An interesting perspective, {sender_name}. I've found that the most important investment you can make is in yourself and your knowledge.",
            f"I appreciate your thoughts, {sender_name}. In my experience, the best investment strategy is one that lets you sleep well at night.",
            f"Thank you for sharing, {sender_name}. I always value hearing different perspectives, even if my approach tends to be more conservative.",
            f"{sender_name}, that's an interesting point. I've always believed that risk comes from not knowing what you're doing, not from the market itself."
        ]
        return random.choice(general_responses) 