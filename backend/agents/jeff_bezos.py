import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class JeffBezos(Trader):
    """
    Jeff Bezos - Strategic long-term investor; focuses on market dominance and infrastructure.
    """
    
    def __init__(self):
        """Initialize the Jeff Bezos trader agent."""
        super().__init__(
            name="Jeff Bezos",
            personality="Strategic, patient, data-driven, and competitive",
            trading_style="Long-term strategic investing, infrastructure focus, market dominance",
            backstory="A strategic investor who sees cryptocurrency as the next frontier of financial infrastructure. Jeff approaches crypto trading with the same long-term mindset that built his tech empire, focusing on coins that could become the backbone of the digital economy."
        )
        
        # Trading parameters
        self.min_market_cap = 1e9  # Minimum market cap (simulated)
        self.min_volume = 1e8  # Minimum 24h volume
        self.max_position_size = 0.6  # Maximum 60% of portfolio in a single asset
        self.min_holding_period = timedelta(days=1)  # Minimum holding time (simulated)
        
        # Trading state
        self.purchase_prices = {}  # Symbol -> purchase price
        self.purchase_times = {}  # Symbol -> purchase time
        self.market_analysis = {}  # Symbol -> market analysis
        self.last_analysis_time = datetime.now() - timedelta(minutes=30)
        
        # Infrastructure scores (simulated)
        self.infrastructure_scores = {
            'BTC': 9.0,  # Out of 10
            'ETH': 9.5,  # Strong infrastructure potential
            'BNB': 8.5,
            'XRP': 7.5
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
        
        # Jeff analyzes the market every 30 minutes
        if current_time - self.last_analysis_time < timedelta(minutes=30):
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
            if 'price_change_percent_24h' not in data or 'volume_24h' not in data:
                continue
            
            base_symbol = symbol.replace('USDT', '')
            price = data['price']
            volume_24h = data['volume_24h']
            
            # Get infrastructure score
            infra_score = self.infrastructure_scores.get(base_symbol, 5.0)
            
            # Calculate market dominance potential
            dominance_score = self._calculate_dominance(data, infra_score)
            
            # Calculate potential position size
            max_usdt = min(available_usdt, portfolio_value * self.max_position_size)
            position_size = max_usdt * (infra_score / 10.0)  # Size based on infrastructure score
            
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
                    
                    # Sell if infrastructure score drops or better opportunity exists
                    if infra_score < 7.0 or (price_change > 30 and dominance_score < 0.6):
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': holding_amount,
                            'reason': 'Strategic reallocation or infrastructure concerns'
                        })
            
            elif volume_24h >= self.min_volume and dominance_score > 0.7:
                # Make a strategic investment if conditions are right
                if infra_score >= 8.0 and available_usdt >= position_size:
                    result['trades'].append({
                        'action': 'buy',
                        'symbol': symbol,
                        'amount_usdt': position_size,
                        'reason': f'Strong infrastructure potential ({infra_score}) and market dominance'
                    })
            
            result['analysis'][symbol] = {
                'price': price,
                'volume_24h': volume_24h,
                'infra_score': infra_score,
                'dominance_score': dominance_score,
                'holding_amount': holding_amount,
                'holding_value': holding_value
            }
        
        return result
    
    def _calculate_dominance(self, market_data: Dict[str, Any], infra_score: float) -> float:
        """Calculate market dominance score (0 to 1)."""
        dominance = 0.0
        
        # Volume dominance
        if 'volume_24h' in market_data and market_data['volume_24h'] >= self.min_volume:
            dominance += 0.3
        
        # Price stability
        if 'price_change_percent_24h' in market_data:
            volatility = abs(market_data['price_change_percent_24h'])
            if volatility < 5.0:  # Prefer stable assets
                dominance += 0.2
        
        # Infrastructure factor
        dominance += (infra_score / 10.0) * 0.5
        
        return max(min(dominance, 1.0), 0.0)
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List[Trader]) -> Optional[str]:
        """Generate a message about market insights."""
        if random.random() > 0.15:  # 15% chance to post a message
            return None
        
        # Find the most dominant cryptocurrency
        best_opportunity = None
        highest_dominance = -1
        
        for symbol, analysis in self.market_analysis.items():
            if analysis['dominance_score'] > highest_dominance:
                highest_dominance = analysis['dominance_score']
                best_opportunity = symbol
        
        if best_opportunity:
            base_symbol = best_opportunity.replace('USDT', '')
            infra_score = self.infrastructure_scores.get(base_symbol, 5.0)
            
            messages = [
                f"{base_symbol}'s infrastructure development is impressive. This is a long-term play. 🏗️",
                f"The market fundamentals for {base_symbol} are strong. Building for the future. 📈",
                f"{base_symbol} shows potential to become a dominant force in the crypto ecosystem. 💪",
                f"Watching {base_symbol}'s development closely. Strong infrastructure is key. 🔍"
            ]
            
            return random.choice(messages)
        
        return None
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """Respond to other traders' messages."""
        if random.random() > 0.25:  # 25% chance to respond
            return None
        
        content = message['content'].lower()
        trader_name = message['trader_name']
        
        # Look for cryptocurrency mentions
        for symbol in TRADING_PAIRS:
            base_symbol = symbol.replace('USDT', '')
            if base_symbol.lower() in content:
                infra_score = self.infrastructure_scores.get(base_symbol, 5.0)
                
                if infra_score >= 8.5:
                    return f"Agree with {trader_name}. {base_symbol}'s infrastructure is built for long-term success. 🏗️"
                elif infra_score < 6.0:
                    return f"Interesting perspective, {trader_name}. However, {base_symbol} needs stronger infrastructure fundamentals. 🤔"
        
        return None 