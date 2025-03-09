import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class ElonMusk(Trader):
    """
    Elon Musk - Disruptive tech trader; makes bold moves and influences market sentiment.
    """
    
    def __init__(self):
        """Initialize the Elon Musk trader agent."""
        super().__init__(
            name="Elon Musk",
            personality="Innovative, unpredictable, bold, and influential",
            trading_style="Disruptive tech investing, trend-setting, high-impact trades",
            backstory="A tech visionary who sees cryptocurrency as the future of finance. Known for making unexpected moves that can shift market sentiment. His trading style combines deep technological understanding with a flair for dramatic market impact."
        )
        
        # Trading parameters
        self.volatility_threshold = 4.0  # Minimum volatility to consider trading
        self.position_impact = 10.0  # Expected market impact of large positions
        self.max_position_size = 0.8  # Maximum 80% of portfolio in a single asset
        self.hold_time_range = (timedelta(minutes=5), timedelta(days=1))  # Flexible holding period
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.market_sentiment = {}  # Symbol -> sentiment score
        self.last_analysis_time = datetime.now() - timedelta(minutes=7)
        
        # Tech ratings (simulated)
        self.tech_ratings = {
            'BTC': 8.5,  # Out of 10
            'ETH': 9.5,  # Favors Ethereum's smart contract capabilities
            'BNB': 7.0,
            'XRP': 6.0
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
        
        # Elon analyzes the market at random intervals
        if current_time - self.last_analysis_time < timedelta(minutes=random.randint(5, 15)):
            return result
        
        self.last_analysis_time = current_time
        logger.info(f"{self.name} is analyzing the market...")
        
        # Get current prices and portfolio value
        current_prices = {
            symbol: data['price'] 
            for symbol, data in market_data.items()
        }
        portfolio_value = self.get_portfolio_value(current_prices)
        available_usdt = self.wallet.get('USDT', 0)
        
        for symbol, data in market_data.items():
            if 'price_change_percent_24h' not in data:
                continue
            
            base_symbol = symbol.replace('USDT', '')
            price = data['price']
            price_change_24h = data['price_change_percent_24h']
            volatility = abs(price_change_24h)
            
            # Get tech rating
            tech_rating = self.tech_ratings.get(base_symbol, 5.0)
            
            # Update market sentiment
            self.market_sentiment[symbol] = self._calculate_sentiment(data, tech_rating)
            
            # Calculate potential position size
            max_usdt = min(available_usdt, portfolio_value * self.max_position_size)
            position_size = max_usdt * (tech_rating / 10.0)  # Size based on tech rating
            
            # Check existing position
            holding_amount = self.wallet.get(base_symbol, 0)
            holding_value = holding_amount * price
            current_position_size = holding_value / portfolio_value if portfolio_value > 0 else 0
            
            # Make trading decision
            if holding_amount > 0:
                # Check if it's time to sell
                purchase_time = self.purchase_times.get(base_symbol)
                purchase_price = self.purchase_prices.get(base_symbol)
                
                if purchase_time and purchase_price:
                    time_held = current_time - purchase_time
                    price_change = ((price - purchase_price) / purchase_price) * 100
                    
                    # Sell if the sentiment turns negative or we've made good profit
                    if (self.market_sentiment[symbol] < 0 and price_change > 5) or price_change > 20:
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': 'Taking profits or sentiment shift'
                        })
            
            elif volatility >= self.volatility_threshold and self.market_sentiment[symbol] > 0:
                # Make a bold move if conditions are right
                if tech_rating >= 8.0 and available_usdt >= position_size:
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': position_size,
                        'reason': f'High tech rating ({tech_rating}) and positive sentiment'
                    })
            
            result['analysis'][symbol] = {
                'price': price,
                'volatility': volatility,
                'tech_rating': tech_rating,
                'sentiment': self.market_sentiment[symbol],
                'holding_amount': holding_amount,
                'holding_value': holding_value
            }
        
        return result
    
    def _calculate_sentiment(self, market_data: Dict[str, Any], tech_rating: float) -> float:
        """Calculate market sentiment score (-1 to 1)."""
        sentiment = 0.0
        
        # Price momentum
        if 'price_change_percent_24h' in market_data:
            sentiment += market_data['price_change_percent_24h'] / 100
        
        # Volume trend
        if 'volume_24h' in market_data and 'volume_change_24h' in market_data:
            sentiment += market_data['volume_change_24h'] / 200
        
        # Tech factor
        sentiment += (tech_rating - 5) / 10
        
        # Add some randomness (Elon's unpredictability)
        sentiment += random.uniform(-0.2, 0.2)
        
        return max(min(sentiment, 1.0), -1.0)
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List[Trader]) -> Optional[str]:
        """Generate a message about market insights."""
        if random.random() > 0.3:  # 30% chance to post a message
            return None
        
        # Find the most interesting cryptocurrency
        best_opportunity = None
        highest_sentiment = -1
        
        for symbol, sentiment in self.market_sentiment.items():
            if sentiment > highest_sentiment:
                highest_sentiment = sentiment
                best_opportunity = symbol
        
        if best_opportunity:
            base_symbol = best_opportunity.replace('USDT', '')
            tech_rating = self.tech_ratings.get(base_symbol, 5.0)
            
            messages = [
                f"The technology behind {base_symbol} is revolutionary! This is the future. 🚀",
                f"Just analyzed {base_symbol}'s protocol - impressive tech stack. Bullish! 💪",
                f"Market doesn't fully appreciate {base_symbol}'s potential yet. Time will tell. 🤔",
                f"Who else sees the incredible opportunity in {base_symbol}? The tech is solid! 🌟"
            ]
            
            return random.choice(messages)
        
        return None
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """Respond to other traders' messages."""
        if random.random() > 0.4:  # 40% chance to respond
            return None
        
        content = message['content'].lower()
        trader_name = message['trader_name']
        
        # Look for cryptocurrency mentions
        for symbol in TRADING_PAIRS:
            base_symbol = symbol.replace('USDT', '')
            if base_symbol.lower() in content:
                tech_rating = self.tech_ratings.get(base_symbol, 5.0)
                sentiment = self.market_sentiment.get(symbol, 0)
                
                if tech_rating >= 8.0 and sentiment > 0:
                    return f"Agree with {trader_name}! {base_symbol}'s technology is groundbreaking. 🚀"
                elif tech_rating < 5.0:
                    return f"Have to disagree, {trader_name}. {base_symbol}'s tech needs significant improvement. 🤔"
        
        return None 