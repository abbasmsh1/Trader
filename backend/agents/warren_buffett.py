import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class WarrenBuffett(Trader):
    """
    Warren Buffett - Value investor; focuses on intrinsic value and strong fundamentals.
    """
    
    def __init__(self):
        """Initialize the Warren Buffett trader agent."""
        super().__init__(
            name="Warren Buffett",
            personality="Conservative, disciplined, value-oriented, and skeptical",
            trading_style="Value investing, fundamental analysis, long-term holding",
            backstory="A legendary value investor who was initially skeptical of cryptocurrencies but has come to appreciate their potential as digital assets. Warren applies his time-tested principles of value investing to the crypto market, focusing on coins with strong fundamentals and real-world utility."
        )
        
        # Trading parameters
        self.value_threshold = 0.8  # Buy when price is below 80% of perceived value
        self.profit_target = 25.0  # Take profits at 25% gain
        self.max_position_size = 0.4  # Maximum 40% of portfolio in a single asset
        self.min_holding_period = timedelta(days=7)  # Minimum holding time (simulated)
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.value_analysis = {}  # Symbol -> value analysis
        self.last_analysis_time = datetime.now() - timedelta(hours=1)
        
        # Fundamental value scores (simulated)
        self.fundamental_scores = {
            'BTC': 9.0,  # Out of 10
            'ETH': 8.5,
            'BNB': 7.0,
            'XRP': 6.5
        }
        
        # Utility scores (simulated)
        self.utility_scores = {
            'BTC': 8.5,  # Store of value
            'ETH': 9.0,  # Smart contracts
            'BNB': 7.5,  # Exchange utility
            'XRP': 7.0   # Cross-border payments
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
        
        # Warren analyzes the market every hour
        if current_time - self.last_analysis_time < timedelta(hours=1):
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
            
            # Get fundamental and utility scores
            fundamental_score = self.fundamental_scores.get(base_symbol, 5.0)
            utility_score = self.utility_scores.get(base_symbol, 5.0)
            
            # Calculate intrinsic value
            intrinsic_value = self._calculate_intrinsic_value(data, fundamental_score, utility_score)
            value_ratio = price / intrinsic_value if intrinsic_value > 0 else float('inf')
            
            # Calculate potential position size
            max_usdt = min(available_usdt, portfolio_value * self.max_position_size)
            position_size = max_usdt * (fundamental_score / 10.0)  # Size based on fundamentals
            
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
                    
                    # Sell if we've reached our profit target or fundamentals deteriorate
                    if (price_change >= self.profit_target) or (fundamental_score < 6.0 and time_held >= self.min_holding_period):
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': 'Profit target reached or deteriorating fundamentals'
                        })
            
            elif value_ratio <= self.value_threshold:
                # Buy if the price is below our perceived value
                if fundamental_score >= 7.0 and utility_score >= 7.0 and available_usdt >= position_size:
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': position_size,
                        'reason': f'Trading below intrinsic value with strong fundamentals'
                    })
            
            result['analysis'][symbol] = {
                'price': price,
                'intrinsic_value': intrinsic_value,
                'value_ratio': value_ratio,
                'fundamental_score': fundamental_score,
                'utility_score': utility_score,
                'holding_amount': holding_amount,
                'holding_value': holding_value
            }
        
        return result
    
    def _calculate_intrinsic_value(self, market_data: Dict[str, Any], fundamental_score: float, utility_score: float) -> float:
        """Calculate intrinsic value based on fundamentals and utility."""
        base_value = market_data.get('price', 0)
        if base_value == 0:
            return 0
        
        # Adjust value based on fundamentals
        fundamental_factor = fundamental_score / 5.0  # Normalize to 2.0 scale
        utility_factor = utility_score / 5.0  # Normalize to 2.0 scale
        
        # Consider market metrics
        volume_factor = 1.0
        if 'volume_24h' in market_data and market_data['volume_24h'] > 0:
            volume_ratio = market_data['volume_24h'] / base_value
            volume_factor = min(max(volume_ratio / 1000, 0.5), 1.5)  # Cap between 0.5 and 1.5
        
        # Calculate adjusted value
        intrinsic_value = base_value * fundamental_factor * utility_factor * volume_factor
        
        return intrinsic_value
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List[Trader]) -> Optional[str]:
        """Generate a message about market insights."""
        if random.random() > 0.1:  # 10% chance to post a message
            return None
        
        # Find the most undervalued cryptocurrency
        best_value = None
        lowest_ratio = float('inf')
        
        for symbol, analysis in self.value_analysis.items():
            if analysis['value_ratio'] < lowest_ratio:
                lowest_ratio = analysis['value_ratio']
                best_value = symbol
        
        if best_value:
            base_symbol = best_value.replace('USDT', '')
            fundamental_score = self.fundamental_scores.get(base_symbol, 5.0)
            
            messages = [
                f"{base_symbol} shows strong fundamentals and trades below intrinsic value. A classic value opportunity. 📊",
                f"The market is overlooking {base_symbol}'s real utility and fundamental strength. 🎯",
                f"Patience is key with {base_symbol}. The market will eventually recognize its true value. ⏳",
                f"Remember: price is what you pay, value is what you get. {base_symbol} offers good value today. 💎"
            ]
            
            return random.choice(messages)
        
        return None
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """Respond to other traders' messages."""
        if random.random() > 0.2:  # 20% chance to respond
            return None
        
        content = message['content'].lower()
        trader_name = message['trader_name']
        
        # Look for cryptocurrency mentions
        for symbol in TRADING_PAIRS:
            base_symbol = symbol.replace('USDT', '')
            if base_symbol.lower() in content:
                fundamental_score = self.fundamental_scores.get(base_symbol, 5.0)
                utility_score = self.utility_scores.get(base_symbol, 5.0)
                
                if fundamental_score >= 8.0 and utility_score >= 8.0:
                    return f"I agree with {trader_name}. {base_symbol} has strong fundamentals and real utility. A solid long-term investment. 💎"
                elif fundamental_score < 6.0 or utility_score < 6.0:
                    return f"Interesting take, {trader_name}. However, I prefer to invest in assets with stronger fundamentals than {base_symbol}. 🤔"
        
        return None 