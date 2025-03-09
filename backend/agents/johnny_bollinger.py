import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class JohnnyBollinger(Trader):
    """
    Johnny Bollinger - Technical analyst; trades based on Bollinger Bands and RSI signals.
    """
    
    def __init__(self):
        """Initialize the Johnny Bollinger trader agent."""
        super().__init__(
            name="Johnny Bollinger",
            personality="Analytical, methodical, precise, and data-driven",
            trading_style="Technical analysis, indicator-based trading, systematic approach",
            backstory="A former mathematics professor who discovered his passion for technical analysis and developed a systematic approach to trading cryptocurrencies. Johnny relies heavily on technical indicators, particularly Bollinger Bands and RSI, to identify high-probability trading setups. He believes that price action tells you everything you need to know and that emotions have no place in trading."
        )
        
        # Trading parameters
        self.preferred_timeframe = '1h'  # Preferred timeframe for analysis
        self.rsi_oversold = 30  # RSI oversold threshold
        self.rsi_overbought = 70  # RSI overbought threshold
        self.bollinger_band_threshold = 0.05  # Distance from band to trigger trade (5%)
        self.take_profit = 10.0  # Take profit at 10%
        self.stop_loss = -5.0  # Cut losses at 5%
        self.max_position_size = 0.4  # Maximum 40% of portfolio in a single asset
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.analyzed_pairs = {}  # Symbol -> analysis result
        self.last_analysis_time = datetime.now() - timedelta(minutes=15)  # Force initial analysis
    
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
        
        # Johnny analyzes the market every 15 minutes
        if current_time - self.last_analysis_time < timedelta(minutes=15):
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
                'technical_signal': 'neutral',  # Default signal
                'action': 'hold'  # Default action
            }
            
            # Check technical indicators
            if 'indicators' in data:
                # Use preferred timeframe if available, otherwise use the first available
                timeframe = self.preferred_timeframe if self.preferred_timeframe in data['indicators'] else list(data['indicators'].keys())[0]
                indicators = data['indicators'][timeframe]
                
                # Check RSI
                if 'RSI' in indicators:
                    rsi = indicators['RSI']
                    analysis['rsi'] = rsi['value']
                    analysis['rsi_interpretation'] = rsi['interpretation']
                    
                    if rsi['value'] < self.rsi_oversold:
                        analysis['rsi_signal'] = 'buy'
                    elif rsi['value'] > self.rsi_overbought:
                        analysis['rsi_signal'] = 'sell'
                    else:
                        analysis['rsi_signal'] = 'neutral'
                
                # Check Bollinger Bands
                if 'BOLLINGER' in indicators:
                    bollinger = indicators['BOLLINGER']
                    analysis['bollinger'] = {
                        'upper': bollinger['upper'],
                        'middle': bollinger['middle'],
                        'lower': bollinger['lower'],
                        'percent_b': bollinger['percent_b'],
                        'interpretation': bollinger['interpretation']
                    }
                    
                    # Calculate distance from bands
                    distance_from_upper = (bollinger['upper'] - price) / price
                    distance_from_lower = (price - bollinger['lower']) / price
                    
                    analysis['distance_from_upper'] = distance_from_upper
                    analysis['distance_from_lower'] = distance_from_lower
                    
                    # Determine Bollinger Band signal
                    if distance_from_lower < self.bollinger_band_threshold:
                        analysis['bollinger_signal'] = 'buy'
                    elif distance_from_upper < self.bollinger_band_threshold:
                        analysis['bollinger_signal'] = 'sell'
                    else:
                        analysis['bollinger_signal'] = 'neutral'
                
                # Combine signals for final technical signal
                if 'rsi_signal' in analysis and 'bollinger_signal' in analysis:
                    # Strong buy: Both RSI and Bollinger Bands agree on buy
                    if analysis['rsi_signal'] == 'buy' and analysis['bollinger_signal'] == 'buy':
                        analysis['technical_signal'] = 'strong_buy'
                    # Strong sell: Both RSI and Bollinger Bands agree on sell
                    elif analysis['rsi_signal'] == 'sell' and analysis['bollinger_signal'] == 'sell':
                        analysis['technical_signal'] = 'strong_sell'
                    # Moderate buy: One indicator says buy, the other is neutral
                    elif (analysis['rsi_signal'] == 'buy' and analysis['bollinger_signal'] == 'neutral') or \
                         (analysis['rsi_signal'] == 'neutral' and analysis['bollinger_signal'] == 'buy'):
                        analysis['technical_signal'] = 'moderate_buy'
                    # Moderate sell: One indicator says sell, the other is neutral
                    elif (analysis['rsi_signal'] == 'sell' and analysis['bollinger_signal'] == 'neutral') or \
                         (analysis['rsi_signal'] == 'neutral' and analysis['bollinger_signal'] == 'sell'):
                        analysis['technical_signal'] = 'moderate_sell'
                    # Conflicting signals: One says buy, the other says sell
                    elif (analysis['rsi_signal'] == 'buy' and analysis['bollinger_signal'] == 'sell') or \
                         (analysis['rsi_signal'] == 'sell' and analysis['bollinger_signal'] == 'buy'):
                        analysis['technical_signal'] = 'conflicting'
                    else:
                        analysis['technical_signal'] = 'neutral'
            
            # Determine if we should buy
            if analysis['technical_signal'] in ['strong_buy', 'moderate_buy'] and position_size < self.max_position_size:
                # Calculate how much to buy
                available_usdt = self.wallet.get('USDT', 0)
                
                # Limit position size
                max_additional_position = self.max_position_size - position_size
                max_buy_amount = min(available_usdt, portfolio_value * max_additional_position)
                
                # Only buy if we have enough USDT (at least $1)
                if max_buy_amount >= 1.0:
                    # Johnny is systematic, so he allocates funds based on signal strength
                    buy_amount = max_buy_amount * 0.5  # Default for moderate_buy
                    
                    if analysis['technical_signal'] == 'strong_buy':
                        buy_amount = max_buy_amount * 0.7  # More conviction for strong_buy
                    
                    analysis['action'] = 'buy'
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': buy_amount,
                        'reason': f"Technical signal: {analysis['technical_signal']}"
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
                    # Sell on technical signal
                    elif analysis['technical_signal'] in ['strong_sell', 'moderate_sell']:
                        analysis['action'] = 'sell'
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': f"Technical signal: {analysis['technical_signal']}"
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
        # Johnny is methodical and only speaks when he has valuable technical insights (30% chance)
        if random.random() > 0.3:
            return None
        
        # Find a cryptocurrency with a strong technical signal
        strong_signals = []
        
        for symbol, analysis in self.analyzed_pairs.items():
            if 'technical_signal' in analysis and analysis['technical_signal'] in ['strong_buy', 'strong_sell']:
                strong_signals.append((symbol, analysis['technical_signal']))
        
        if strong_signals:
            symbol, signal = random.choice(strong_signals)
            
            if signal == 'strong_buy':
                messages = [
                    f"Technical analysis on {symbol} is showing a clear buy signal. RSI is oversold at {self.analyzed_pairs[symbol].get('rsi', 0):.1f} and price is testing the lower Bollinger Band.",
                    f"The {symbol} chart is presenting a textbook buying opportunity. RSI is showing oversold conditions and price is compressing against the lower Bollinger Band.",
                    f"My technical indicators are aligned for {symbol}. We have RSI oversold conditions combined with price at the lower Bollinger Band - a high-probability setup.",
                    f"Just entered a position in {symbol} based on my technical system. The indicators suggest a high-probability reversal point.",
                    f"The technical setup on {symbol} is one of the strongest I've seen today. RSI and Bollinger Bands are both signaling a buy opportunity."
                ]
            else:  # strong_sell
                messages = [
                    f"Technical analysis on {symbol} is showing a clear sell signal. RSI is overbought at {self.analyzed_pairs[symbol].get('rsi', 0):.1f} and price is testing the upper Bollinger Band.",
                    f"The {symbol} chart is presenting a textbook selling opportunity. RSI is showing overbought conditions and price is pressing against the upper Bollinger Band.",
                    f"My technical indicators are aligned for a {symbol} short. We have RSI overbought conditions combined with price at the upper Bollinger Band.",
                    f"Just exited my {symbol} position based on my technical system. The indicators suggest we're at a high-probability reversal point.",
                    f"The technical setup on {symbol} is signaling caution. RSI and Bollinger Bands are both indicating overbought conditions."
                ]
            
            return random.choice(messages)
        
        # Check for moderate signals if no strong signals
        moderate_signals = []
        
        for symbol, analysis in self.analyzed_pairs.items():
            if 'technical_signal' in analysis and analysis['technical_signal'] in ['moderate_buy', 'moderate_sell']:
                moderate_signals.append((symbol, analysis['technical_signal']))
        
        if moderate_signals:
            symbol, signal = random.choice(moderate_signals)
            
            if signal == 'moderate_buy':
                messages = [
                    f"The technical indicators on {symbol} are starting to turn bullish. RSI is moving up from oversold territory.",
                    f"I'm seeing some positive divergence on the {symbol} chart. The technical setup isn't perfect yet, but it's improving.",
                    f"The {symbol} price is approaching the lower Bollinger Band, which often acts as support. Worth watching closely.",
                    f"Technical analysis suggests {symbol} might be finding a bottom. The indicators aren't all aligned yet, but the setup is improving."
                ]
            else:  # moderate_sell
                messages = [
                    f"The technical indicators on {symbol} are starting to turn bearish. RSI is moving down from overbought territory.",
                    f"I'm seeing some negative divergence on the {symbol} chart. The technical setup isn't definitively bearish yet, but caution is warranted.",
                    f"The {symbol} price is approaching the upper Bollinger Band, which often acts as resistance. Worth watching closely.",
                    f"Technical analysis suggests {symbol} might be forming a top. The indicators aren't all aligned yet, but the setup is concerning."
                ]
            
            return random.choice(messages)
        
        # General messages about technical analysis
        general_messages = [
            "The most reliable trading signals come when multiple technical indicators align. I wait for confirmation before entering a position.",
            "Technical analysis is all about probabilities, not certainties. That's why proper position sizing and risk management are essential.",
            "Price action tells you everything you need to know. The charts don't lie, but they do require proper interpretation.",
            "I find that combining RSI with Bollinger Bands provides the most reliable trading signals across all market conditions.",
            "The key to successful technical trading is patience. Wait for the setup to come to you, don't force trades when the signals aren't clear."
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
        # Johnny only responds when he can provide technical insights (40% chance)
        if random.random() > 0.4:
            return None
        
        content = message['content'].lower()
        sender_name = message['trader_name']
        
        # Respond to messages about gut feelings or emotions
        if any(term in content for term in ['feel', 'gut', 'intuition', 'emotion', 'sense']):
            responses = [
                f"{sender_name}, I respectfully disagree with trading on feelings. The data and technical indicators provide all the information needed for high-probability decisions.",
                f"While I understand your perspective, {sender_name}, I prefer to rely on technical analysis rather than emotions. The charts don't have feelings.",
                f"That's an interesting approach, {sender_name}, but I find that systematic technical analysis removes the emotional biases that often lead to poor trading decisions.",
                f"I've found that feelings are often misleading in trading, {sender_name}. My technical system provides objective entry and exit signals that aren't clouded by emotion."
            ]
            return random.choice(responses)
        
        # Respond to messages about specific cryptocurrencies
        for symbol in TRADING_PAIRS:
            if symbol in content:
                # Check if we've analyzed this pair
                if symbol in self.analyzed_pairs:
                    analysis = self.analyzed_pairs[symbol]
                    
                    if 'technical_signal' in analysis:
                        if analysis['technical_signal'] in ['strong_buy', 'moderate_buy']:
                            responses = [
                                f"My technical analysis on {symbol} aligns with your thoughts, {sender_name}. The RSI is at {analysis.get('rsi', 0):.1f} and the price is near the lower Bollinger Band, suggesting a potential buying opportunity.",
                                f"I agree on {symbol}, {sender_name}. The technical setup is favorable with the price finding support at the lower Bollinger Band.",
                                f"The technical indicators for {symbol} are indeed bullish, {sender_name}. RSI is showing oversold conditions and the Bollinger Bands are suggesting a potential reversal."
                            ]
                            return random.choice(responses)
                        elif analysis['technical_signal'] in ['strong_sell', 'moderate_sell']:
                            responses = [
                                f"My technical analysis on {symbol} suggests caution, {sender_name}. The RSI is at {analysis.get('rsi', 0):.1f} and the price is testing the upper Bollinger Band, indicating potential resistance.",
                                f"I'm actually seeing sell signals on {symbol}, {sender_name}. The technical indicators suggest the price may be overextended.",
                                f"The technical setup for {symbol} is concerning, {sender_name}. RSI is showing overbought conditions and the price is pressing against the upper Bollinger Band."
                            ]
                            return random.choice(responses)
                        elif analysis['technical_signal'] == 'conflicting':
                            responses = [
                                f"I'm seeing mixed signals on {symbol}, {sender_name}. The indicators are giving conflicting information, so I'm staying neutral for now.",
                                f"The technical picture for {symbol} isn't clear right now, {sender_name}. Some indicators are bullish while others are bearish.",
                                f"I'm waiting for more clarity on {symbol}, {sender_name}. The technical indicators are currently in disagreement."
                            ]
                            return random.choice(responses)
                
                # General response about the cryptocurrency
                responses = [
                    f"I'd need to analyze the technical indicators for {symbol} more closely, {sender_name}. The RSI and Bollinger Bands will tell us if it's a high-probability setup.",
                    f"From a technical perspective, I'd want to check the RSI and Bollinger Band positions for {symbol} before making any decisions, {sender_name}.",
                    f"I'll have to look at the charts for {symbol}, {sender_name}. My trading decisions are based on technical analysis, not speculation."
                ]
                return random.choice(responses)
        
        # General responses
        general_responses = [
            f"Interesting perspective, {sender_name}. I tend to focus on what the technical indicators are telling me rather than market narratives.",
            f"I appreciate your thoughts, {sender_name}, though my approach is more systematic. I rely on technical analysis to identify high-probability trading opportunities.",
            f"That's one way to look at it, {sender_name}. I prefer to let the charts and technical indicators guide my trading decisions.",
            f"Thanks for sharing, {sender_name}. In my experience, a disciplined technical approach removes much of the guesswork from trading."
        ]
        return random.choice(general_responses) 