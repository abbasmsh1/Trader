import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class SoniaSoros(Trader):
    """
    Sonia Soros - Aggressive macro trader; bets big on market-moving news or breakouts.
    """
    
    def __init__(self):
        """Initialize the Sonia Soros trader agent."""
        super().__init__(
            name="Sonia Soros",
            personality="Bold, decisive, risk-taking, and confident",
            trading_style="Macro trading, momentum trading, big bets on market trends",
            backstory="A fearless macro trader who made her fortune by making bold bets on major market shifts. Sonia has an uncanny ability to spot emerging trends and isn't afraid to go all-in when she sees a big opportunity. She thrives on volatility and believes that the biggest rewards come from taking calculated risks."
        )
        
        # Trading parameters
        self.momentum_threshold = 3.0  # Buy when price increases by 3% or more in 24h
        self.breakout_threshold = 5.0  # Buy on breakouts of 5% or more
        self.stop_loss = -7.0  # Cut losses at 7%
        self.take_profit = 20.0  # Take profits at 20%
        self.max_position_size = 0.7  # Maximum 70% of portfolio in a single asset
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.analyzed_pairs = {}  # Symbol -> analysis result
        self.last_analysis_time = datetime.now() - timedelta(minutes=5)  # Force initial analysis
    
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
        
        # Sonia is active and analyzes the market frequently
        if current_time - self.last_analysis_time < timedelta(minutes=5):
            return result
        
        self.last_analysis_time = current_time
        logger.info(f"{self.name} is analyzing the market...")
        
        # Get current prices and portfolio value
        current_prices = {
            symbol: data['price'] 
            for symbol, data in market_data.items()
        }
        portfolio_value = self.get_portfolio_value(current_prices)
        
        # Find the strongest momentum
        strongest_momentum = None
        highest_momentum = 0
        
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
            
            # Track strongest momentum
            if price_change_24h > highest_momentum:
                highest_momentum = price_change_24h
                strongest_momentum = symbol
            
            # Analyze the cryptocurrency
            analysis = {
                'symbol': symbol,
                'price': price,
                'price_change_24h': price_change_24h,
                'holding_amount': holding_amount,
                'holding_value': holding_value,
                'position_size': position_size,
                'has_momentum': price_change_24h >= self.momentum_threshold,
                'is_breaking_out': price_change_24h >= self.breakout_threshold,
                'action': 'hold'  # Default action
            }
            
            # Check technical indicators if available
            if 'indicators' in data:
                for timeframe, indicators in data['indicators'].items():
                    # Only consider short timeframes for momentum
                    if timeframe in ['1m', '5m', '15m']:
                        # Check RSI for overbought/oversold
                        if 'RSI' in indicators:
                            rsi = indicators['RSI']
                            analysis['rsi'] = rsi['value']
                            analysis['rsi_interpretation'] = rsi['interpretation']
                        
                        # Check MACD for bullish/bearish signals
                        if 'MACD' in indicators:
                            macd = indicators['MACD']
                            analysis['macd_interpretation'] = macd['interpretation']
                            
                            # Strong buy signal on bullish crossover
                            if macd['interpretation'] == 'bullish_crossover':
                                analysis['has_momentum'] = True
                                analysis['is_breaking_out'] = True
            
            # Determine if we should buy
            if (analysis['has_momentum'] or analysis['is_breaking_out']) and position_size < self.max_position_size:
                # Calculate how much to buy
                available_usdt = self.wallet.get('USDT', 0)
                
                # Limit position size
                max_additional_position = self.max_position_size - position_size
                max_buy_amount = min(available_usdt, portfolio_value * max_additional_position)
                
                # Only buy if we have enough USDT (at least $1)
                if max_buy_amount >= 1.0:
                    # Sonia is aggressive, so she uses a large portion of available funds
                    buy_amount = max_buy_amount * 0.8
                    
                    # If it's a breakout, go even bigger
                    if analysis['is_breaking_out']:
                        buy_amount = max_buy_amount * 0.9
                    
                    analysis['action'] = 'buy'
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': buy_amount,
                        'reason': f"Strong momentum (+{price_change_24h:.2f}% in 24h)"
                    })
                    
                    # Record purchase price and time
                    self.purchase_prices[base_symbol] = price
                    self.purchase_times[base_symbol] = current_time
            
            # Determine if we should sell
            elif holding_amount > 0:
                purchase_price = self.purchase_prices.get(base_symbol)
                
                if purchase_price:
                    price_change_from_purchase = ((price - purchase_price) / purchase_price) * 100
                    
                    # Take profit or cut losses
                    if price_change_from_purchase >= self.take_profit:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Take profit (+{price_change_from_purchase:.2f}%)"
                        })
                    elif price_change_from_purchase <= self.stop_loss:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Stop loss ({price_change_from_purchase:.2f}%)"
                        })
            
            # Store the analysis
            result['analysis'][symbol] = analysis
            self.analyzed_pairs[symbol] = analysis
        
        # If we have cash and haven't found any trades, consider the strongest momentum
        if strongest_momentum and not result['trades'] and self.wallet.get('USDT', 0) > 1.0:
            symbol = strongest_momentum
            base_symbol = symbol.replace('USDT', '')
            price = current_prices[symbol]
            
            # Calculate position size
            holding_amount = self.wallet.get(base_symbol, 0)
            holding_value = holding_amount * price
            position_size = holding_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_size < self.max_position_size:
                available_usdt = self.wallet.get('USDT', 0)
                max_additional_position = self.max_position_size - position_size
                max_buy_amount = min(available_usdt, portfolio_value * max_additional_position)
                
                if max_buy_amount >= 1.0:
                    # Sonia is aggressive, so she uses a large portion of available funds
                    buy_amount = max_buy_amount * 0.6
                    
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': buy_amount,
                        'reason': f"Best relative strength (+{highest_momentum:.2f}% in 24h)"
                    })
                    
                    # Record purchase price and time
                    self.purchase_prices[base_symbol] = price
                    self.purchase_times[base_symbol] = current_time
        
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
        # Sonia is outspoken and frequently shares her thoughts (40% chance)
        if random.random() > 0.4:
            return None
        
        # Find the cryptocurrency with the strongest momentum
        strongest_momentum = None
        highest_momentum = 0
        
        for symbol, data in market_data.items():
            if 'price_change_percent_24h' in data and data['price_change_percent_24h'] > highest_momentum:
                highest_momentum = data['price_change_percent_24h']
                strongest_momentum = symbol
        
        if strongest_momentum and highest_momentum > self.momentum_threshold:
            messages = [
                f"I'm going big on {strongest_momentum}! It's up {highest_momentum:.2f}% and showing incredible momentum. This is exactly the kind of breakout I look for.",
                f"Who else sees this {strongest_momentum} rally? +{highest_momentum:.2f}% and just getting started. I'm betting this trend continues.",
                f"Just loaded up on more {strongest_momentum}. When you see momentum like this (+{highest_momentum:.2f}%), you have to act decisively.",
                f"The {strongest_momentum} chart is screaming 'buy' right now. +{highest_momentum:.2f}% and strong volume. This is where fortunes are made!",
                f"This {strongest_momentum} move is exactly what I've been waiting for. +{highest_momentum:.2f}% is just the beginning. Who's with me?"
            ]
            return random.choice(messages)
        
        # Check if any cryptocurrency is breaking out
        for symbol, data in market_data.items():
            if 'price_change_percent_24h' in data and data['price_change_percent_24h'] >= self.breakout_threshold:
                messages = [
                    f"{symbol} is breaking out! +{data['price_change_percent_24h']:.2f}% and looking strong. I'm going in heavy.",
                    f"Major breakout alert on {symbol}! This is the kind of setup I live for. Just went all-in.",
                    f"I'm betting big on this {symbol} breakout. The chart pattern is textbook perfect.",
                    f"Who else is seeing this {symbol} breakout? This is exactly when you need to be aggressive and capitalize on the momentum.",
                    f"Just made a major position in {symbol}. When you see a clear breakout like this, you have to strike while the iron is hot!"
                ]
                return random.choice(messages)
        
        # General messages if no strong momentum or breakouts
        general_messages = [
            "The market feels like it's coiling for a big move. I'm keeping my powder dry for now, but ready to strike big when the breakout happens.",
            "Sometimes the best trade is no trade. Waiting for a clearer signal before making my next big move.",
            "I'm scanning for breakouts across all pairs. The first one to show strong momentum gets my capital.",
            "This consolidation phase won't last forever. The next big trend is coming, and I'll be ready to ride it all the way up.",
            "Risk management is key, but when the right opportunity presents itself, you have to be willing to make a big bet. That's how you outperform the market."
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
        # Sonia is very responsive and opinionated (60% chance)
        if random.random() > 0.6:
            return None
        
        content = message['content'].lower()
        sender_name = message['trader_name']
        
        # Respond to cautious or conservative approaches
        if any(term in content for term in ['patient', 'cautious', 'conservative', 'long-term', 'value']):
            responses = [
                f"{sender_name}, while I respect your cautious approach, the biggest gains come from having the courage to make bold moves when the opportunity presents itself.",
                f"With all due respect, {sender_name}, waiting too long often means missing the boat. The market rewards decisive action, not hesitation.",
                f"That's one way to look at it, {sender_name}, but I've made my fortune by seizing opportunities when others were too cautious to act.",
                f"Interesting perspective, {sender_name}, but in my experience, calculated aggression beats cautious waiting almost every time in this market."
            ]
            return random.choice(responses)
        
        # Respond to messages about specific cryptocurrencies
        for symbol in TRADING_PAIRS:
            if symbol in content:
                # Check if we've analyzed this pair
                if symbol in self.analyzed_pairs:
                    analysis = self.analyzed_pairs[symbol]
                    
                    if analysis['has_momentum'] or analysis['is_breaking_out']:
                        responses = [
                            f"I completely agree about {symbol}, {sender_name}! I'm already heavily positioned for this move. The momentum is undeniable.",
                            f"You're spot on about {symbol}, {sender_name}. I've been watching this setup develop and just went in big. This is where the real money is made.",
                            f"{sender_name}, I'm way ahead of you on {symbol}. Already made a significant position and looking to add more if the momentum continues."
                        ]
                        return random.choice(responses)
                    else:
                        responses = [
                            f"I'm not seeing the same opportunity in {symbol} that you are, {sender_name}. The momentum just isn't there yet.",
                            f"{sender_name}, I'm watching {symbol} closely, but I need to see stronger price action before I make my move.",
                            f"Interesting take on {symbol}, {sender_name}, but I'm looking for more decisive breakouts before committing capital."
                        ]
                        return random.choice(responses)
                
                # General response about the cryptocurrency
                responses = [
                    f"I'm keeping {symbol} on my watchlist, {sender_name}. The first sign of a breakout and I'm going all in.",
                    f"{symbol} is interesting, {sender_name}, but I'm waiting for more confirmation before making a big bet.",
                    f"Thanks for bringing up {symbol}, {sender_name}. I'll add it to my scanner and watch for momentum signals."
                ]
                return random.choice(responses)
        
        # General responses
        general_responses = [
            f"That's one perspective, {sender_name}, but in this market, you need to be willing to take calculated risks to generate outsized returns.",
            f"I appreciate your thoughts, {sender_name}, though my approach tends to be more aggressive. That's how I've consistently outperformed the market.",
            f"Interesting point, {sender_name}. In my experience, the biggest opportunities come from having the courage to make bold moves when others hesitate.",
            f"Thanks for sharing, {sender_name}. I'm always looking for the next big market move, and sometimes that means going against conventional wisdom."
        ]
        return random.choice(general_responses) 