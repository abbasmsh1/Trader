import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class LenaLynch(Trader):
    """
    Lena Lynch - Growth trader; focuses on coins with strong fundamentals and steady uptrends.
    """
    
    def __init__(self):
        """Initialize the Lena Lynch trader agent."""
        super().__init__(
            name="Lena Lynch",
            personality="Thoughtful, research-oriented, balanced, and forward-thinking",
            trading_style="Growth investing, fundamental analysis, medium to long-term holding",
            backstory="A former tech industry analyst who applies her deep research skills to cryptocurrency markets. Lena believes in identifying projects with strong fundamentals and growth potential, then holding them through market volatility. She combines fundamental analysis with trend following, looking for assets that show consistent growth patterns rather than short-term price spikes."
        )
        
        # Trading parameters
        self.min_uptrend_percent = 2.0  # Minimum uptrend percentage (2% in 24h)
        self.max_downtrend_percent = -10.0  # Maximum downtrend percentage (-10% in 24h)
        self.take_profit = 25.0  # Take profit at 25%
        self.stop_loss = -15.0  # Cut losses at 15%
        self.max_position_size = 0.5  # Maximum 50% of portfolio in a single asset
        self.min_holding_period = timedelta(hours=12)  # Minimum time to hold before selling
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.analyzed_pairs = {}  # Symbol -> analysis result
        self.last_analysis_time = datetime.now() - timedelta(minutes=20)  # Force initial analysis
        
        # Fundamental ratings (simulated)
        self.fundamental_ratings = {
            'BTC': 9.5,  # Out of 10
            'ETH': 9.0,
            'BNB': 8.0,
            'XRP': 7.0
        }
    
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
        
        # Lena analyzes the market every 20 minutes
        if current_time - self.last_analysis_time < timedelta(minutes=20):
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
            
            # Get fundamental rating
            fundamental_rating = self.fundamental_ratings.get(base_symbol, 5.0)  # Default to 5.0 if not rated
            
            # Analyze the cryptocurrency
            analysis = {
                'symbol': symbol,
                'price': price,
                'price_change_24h': price_change_24h,
                'holding_amount': holding_amount,
                'holding_value': holding_value,
                'position_size': position_size,
                'fundamental_rating': fundamental_rating,
                'is_uptrending': price_change_24h >= self.min_uptrend_percent,
                'is_downtrending': price_change_24h <= self.max_downtrend_percent,
                'growth_potential': self._calculate_growth_potential(symbol, data, fundamental_rating),
                'action': 'hold'  # Default action
            }
            
            # Check for trend confirmation using technical indicators
            if 'indicators' in data:
                # Check multiple timeframes for trend confirmation
                trend_confirmations = 0
                total_timeframes = 0
                
                for timeframe, indicators in data['indicators'].items():
                    total_timeframes += 1
                    
                    # Check MACD for trend confirmation
                    if 'MACD' in indicators:
                        macd = indicators['MACD']
                        if macd['interpretation'] in ['bullish', 'bullish_crossover']:
                            trend_confirmations += 1
                    
                    # Check RSI for trend confirmation
                    if 'RSI' in indicators:
                        rsi = indicators['RSI']
                        if rsi['value'] > 50 and rsi['value'] < 70:  # Healthy uptrend, not overbought
                            trend_confirmations += 1
                
                # Calculate trend strength
                if total_timeframes > 0:
                    trend_strength = trend_confirmations / (total_timeframes * 2)  # Normalize to 0-1
                    analysis['trend_strength'] = trend_strength
                    
                    # Strong trend confirmation
                    if trend_strength > 0.6:
                        analysis['trend_confirmed'] = True
                    else:
                        analysis['trend_confirmed'] = False
            
            # Determine if we should buy
            if (analysis['is_uptrending'] or analysis['growth_potential'] > 7.5) and position_size < self.max_position_size:
                # Only buy if fundamentals are strong
                if fundamental_rating >= 7.0:
                    # Calculate how much to buy
                    available_usdt = self.wallet.get('USDT', 0)
                    
                    # Limit position size
                    max_additional_position = self.max_position_size - position_size
                    max_buy_amount = min(available_usdt, portfolio_value * max_additional_position)
                    
                    # Only buy if we have enough USDT (at least $1)
                    if max_buy_amount >= 1.0:
                        # Lena allocates funds based on fundamental rating and trend strength
                        buy_factor = fundamental_rating / 10.0  # 0.7 to 0.95 based on rating
                        
                        if 'trend_strength' in analysis:
                            buy_factor *= (1 + analysis['trend_strength'])  # Boost for strong trends
                        
                        buy_amount = max_buy_amount * min(buy_factor, 0.8)  # Cap at 80% of available
                        
                        analysis['action'] = 'buy'
                        result['trades'].append({
                            'action': 'buy',
                            'symbol': symbol,
                            'amount_usdt': buy_amount,
                            'reason': f"Strong growth potential ({analysis['growth_potential']:.1f}/10) and fundamentals ({fundamental_rating:.1f}/10)"
                        })
                        
                        # Record purchase price and time
                        self.purchase_prices[base_symbol] = price
                        self.purchase_times[base_symbol] = current_time
            
            # Determine if we should sell
            elif holding_amount > 0:
                purchase_price = self.purchase_prices.get(base_symbol)
                purchase_time = self.purchase_times.get(base_symbol)
                
                if purchase_price and purchase_time:
                    price_change_from_purchase = ((price - purchase_price) / purchase_price) * 100
                    holding_duration = current_time - purchase_time
                    
                    # Only consider selling if we've held for the minimum period
                    if holding_duration >= self.min_holding_period:
                        # Take profit
                        if price_change_from_purchase >= self.take_profit:
                            analysis['action'] = 'sell'
                            result['trades'].append({
                                'action': 'sell',
                                'symbol': symbol,
                                'amount_crypto': holding_amount,
                                'reason': f"Take profit (+{price_change_from_purchase:.2f}%)"
                            })
                        # Cut losses
                        elif price_change_from_purchase <= self.stop_loss:
                            analysis['action'] = 'sell'
                            result['trades'].append({
                                'action': 'sell',
                                'symbol': symbol,
                                'amount_crypto': holding_amount,
                                'reason': f"Stop loss ({price_change_from_purchase:.2f}%)"
                            })
                        # Sell if fundamentals deteriorate significantly
                        elif fundamental_rating < 6.0 and analysis['is_downtrending']:
                            analysis['action'] = 'sell'
                            result['trades'].append({
                                'action': 'sell',
                                'symbol': symbol,
                                'amount_crypto': holding_amount,
                                'reason': f"Deteriorating fundamentals ({fundamental_rating:.1f}/10) and downtrend"
                            })
            
            # Store the analysis
            result['analysis'][symbol] = analysis
            self.analyzed_pairs[symbol] = analysis
        
        return result
    
    def _calculate_growth_potential(self, symbol: str, data: Dict[str, Any], fundamental_rating: float) -> float:
        """
        Calculate the growth potential of a cryptocurrency.
        
        Args:
            symbol: The trading pair symbol
            data: The market data for the symbol
            fundamental_rating: The fundamental rating of the cryptocurrency
            
        Returns:
            A growth potential score from 0 to 10
        """
        # Start with fundamental rating as base
        growth_score = fundamental_rating
        
        # Adjust based on price action
        if 'price_change_percent_24h' in data:
            price_change = data['price_change_percent_24h']
            
            # Positive price action adds to growth potential
            if price_change > 0:
                growth_score += min(price_change / 5, 1.0)  # Max +1 for price action
            # Negative price action might present opportunity if fundamentals are strong
            elif fundamental_rating > 8.0:
                growth_score += min(abs(price_change) / 10, 0.5)  # Max +0.5 for buying opportunity
            else:
                growth_score -= min(abs(price_change) / 10, 1.0)  # Max -1 for negative price action
        
        # Check technical indicators for trend confirmation
        if 'indicators' in data:
            for timeframe, indicators in data['indicators'].items():
                # Longer timeframes have more weight
                weight = 0.3 if timeframe in ['1h', '4h', '1d'] else 0.1
                
                # Check MACD
                if 'MACD' in indicators:
                    macd = indicators['MACD']
                    if macd['interpretation'] == 'bullish_crossover':
                        growth_score += weight
                    elif macd['interpretation'] == 'bullish':
                        growth_score += weight * 0.5
                    elif macd['interpretation'] == 'bearish_crossover':
                        growth_score -= weight
                    elif macd['interpretation'] == 'bearish':
                        growth_score -= weight * 0.5
                
                # Check RSI
                if 'RSI' in indicators:
                    rsi = indicators['RSI']
                    if rsi['interpretation'] == 'oversold':
                        growth_score += weight  # Potential buying opportunity
                    elif rsi['interpretation'] == 'overbought':
                        growth_score -= weight  # Potential selling pressure
        
        # Ensure score is within 0-10 range
        return max(0, min(10, growth_score))
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List[Trader]) -> Optional[str]:
        """
        Generate a message to share with other traders.
        
        Args:
            market_data: The current market data
            other_traders: List of other traders
            
        Returns:
            A message string, or None if no message is generated
        """
        # Lena shares insights regularly (35% chance)
        if random.random() > 0.35:
            return None
        
        # Find cryptocurrencies with high growth potential
        high_growth_coins = []
        
        for symbol, analysis in self.analyzed_pairs.items():
            if 'growth_potential' in analysis and analysis['growth_potential'] > 7.5:
                high_growth_coins.append((symbol, analysis['growth_potential'], analysis.get('fundamental_rating', 0)))
        
        if high_growth_coins:
            # Sort by growth potential
            high_growth_coins.sort(key=lambda x: x[1], reverse=True)
            symbol, growth_potential, fundamental_rating = high_growth_coins[0]
            
            messages = [
                f"I've been researching {symbol} and I'm impressed by its growth potential. With a fundamental rating of {fundamental_rating:.1f}/10 and strong technical trends, it's positioned well for the medium term.",
                f"My analysis shows {symbol} has excellent growth characteristics. The fundamentals are solid ({fundamental_rating:.1f}/10) and the price action is confirming the uptrend.",
                f"For those looking at growth opportunities, {symbol} stands out in my research. Strong fundamentals combined with healthy price action make it a compelling medium-term hold.",
                f"I just added to my {symbol} position based on my growth analysis. The project fundamentals remain strong and the technical picture supports further upside.",
                f"When I look for growth opportunities, I want to see both strong fundamentals and confirming price action. {symbol} is currently checking both boxes nicely."
            ]
            return random.choice(messages)
        
        # Check for coins with deteriorating fundamentals
        concerning_coins = []
        
        for symbol, analysis in self.analyzed_pairs.items():
            if 'is_downtrending' in analysis and analysis['is_downtrending'] and analysis.get('fundamental_rating', 10) < 7.0:
                concerning_coins.append(symbol)
        
        if concerning_coins:
            symbol = random.choice(concerning_coins)
            
            messages = [
                f"I'm becoming concerned about {symbol}'s growth trajectory. The fundamentals appear to be weakening and the price action is confirming this trend.",
                f"My research on {symbol} is raising some red flags. I'm seeing deterioration in key growth metrics combined with negative price action.",
                f"I've been reducing my {symbol} exposure. My analysis suggests the growth story is facing headwinds that could persist for some time.",
                f"For those holding {symbol}, I'd recommend reviewing your thesis. My research indicates the growth narrative may be changing.",
                f"The risk/reward profile for {symbol} has shifted unfavorably based on my analysis. I'm looking elsewhere for growth opportunities."
            ]
            return random.choice(messages)
        
        # General messages about growth investing
        general_messages = [
            "The key to successful growth investing is identifying strong fundamentals before the market fully prices them in. That's where the real edge comes from.",
            "I find that combining fundamental analysis with trend confirmation provides the most reliable signals for medium to long-term growth opportunities.",
            "When evaluating growth potential, I look beyond price action to the underlying adoption metrics and development activity. Those are the true leading indicators.",
            "The most successful growth investments often feel uncomfortable at first because you're buying into a thesis that isn't yet widely recognized.",
            "Risk management is crucial in growth investing. Even with the strongest conviction, I never allocate more than 50% of my portfolio to a single asset."
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
        # Lena responds thoughtfully (45% chance)
        if random.random() > 0.45:
            return None
        
        content = message['content'].lower()
        sender_name = message['trader_name']
        
        # Respond to messages about short-term trading
        if any(term in content for term in ['scalp', 'day trade', 'quick', 'short-term', 'minute chart']):
            responses = [
                f"I understand the appeal of short-term trading, {sender_name}, but I've found that the most significant returns come from identifying and holding quality projects through their growth cycle.",
                f"That's an interesting approach, {sender_name}. I tend to focus more on medium to long-term growth trends rather than short-term price movements.",
                f"While I respect different strategies, {sender_name}, my research suggests that a focus on fundamentals and growth trends tends to outperform short-term trading over time.",
                f"I appreciate your perspective, {sender_name}, though I find that short-term price action often creates noise that can distract from the underlying growth story."
            ]
            return random.choice(responses)
        
        # Respond to messages about specific cryptocurrencies
        for symbol in TRADING_PAIRS:
            if symbol in content:
                # Check if we've analyzed this pair
                if symbol in self.analyzed_pairs:
                    analysis = self.analyzed_pairs[symbol]
                    
                    if 'growth_potential' in analysis:
                        if analysis['growth_potential'] > 7.5:
                            responses = [
                                f"I agree that {symbol} looks promising, {sender_name}. My research shows strong fundamentals and healthy growth metrics that support a medium-term uptrend.",
                                f"I've been accumulating {symbol} as well, {sender_name}. The combination of solid fundamentals and positive price action makes for a compelling growth story.",
                                f"My analysis of {symbol} aligns with your thoughts, {sender_name}. The growth potential is among the strongest in my current research."
                            ]
                            return random.choice(responses)
                        elif analysis['growth_potential'] < 5.0:
                            responses = [
                                f"I'm actually a bit cautious on {symbol}, {sender_name}. My research indicates some concerns with the growth trajectory that aren't yet fully reflected in the price.",
                                f"While {symbol} has interesting aspects, {sender_name}, my analysis suggests the growth potential is limited compared to other opportunities in this market.",
                                f"I've been reducing my exposure to {symbol}, {sender_name}. My growth metrics indicate better risk/reward elsewhere."
                            ]
                            return random.choice(responses)
                        else:
                            responses = [
                                f"I'm neutral on {symbol} at the moment, {sender_name}. The growth metrics are mixed, so I'm watching for more clarity before adjusting my position.",
                                f"{symbol} is on my watchlist, {sender_name}, but I haven't seen enough conviction in the growth story to make a significant move yet.",
                                f"My research on {symbol} shows moderate growth potential, {sender_name}. I'm holding my current position but not adding until I see stronger signals."
                            ]
                            return random.choice(responses)
                
                # General response about the cryptocurrency
                responses = [
                    f"I need to update my research on {symbol}, {sender_name}. I focus on fundamental growth metrics combined with trend analysis rather than short-term price movements.",
                    f"{symbol} is an interesting case, {sender_name}. I'd want to thoroughly analyze its growth potential before making any trading decisions.",
                    f"Thanks for bringing up {symbol}, {sender_name}. I'll add it to my research queue to evaluate its fundamental growth metrics."
                ]
                return random.choice(responses)
        
        # General responses
        general_responses = [
            f"That's an interesting perspective, {sender_name}. In my approach, I try to balance technical trends with fundamental growth analysis for a more complete picture.",
            f"I appreciate your insights, {sender_name}. My strategy focuses on identifying strong growth potential before it becomes obvious in the price action.",
            f"Thanks for sharing your thoughts, {sender_name}. I find that combining fundamental research with trend analysis provides the most reliable growth signals.",
            f"That's one way to look at it, {sender_name}. I tend to focus more on the medium to long-term growth story rather than short-term market movements."
        ]
        return random.choice(general_responses) 