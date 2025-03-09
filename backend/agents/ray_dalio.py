import logging
from typing import Dict, List, Any, Optional
import random
from datetime import datetime, timedelta

from ..models.trader import Trader
from ..config import TRADING_PAIRS

logger = logging.getLogger(__name__)


class RayDalio(Trader):
    """
    Ray Dalio - Macro investor; focuses on economic cycles and risk parity.
    """
    
    def __init__(self):
        """Initialize the Ray Dalio trader agent."""
        super().__init__(
            name="Ray Dalio",
            personality="Systematic, macro-focused, and risk-conscious",
            trading_style="All-weather portfolio strategy, macro trend following",
            backstory="A legendary macro investor who applies economic principles to crypto markets. Known for his systematic approach and focus on understanding market cycles. Believes in maintaining a balanced portfolio that can weather any economic environment."
        )
        
        # Trading parameters
        self.risk_allocation = 0.25  # Equal risk allocation (25% per asset)
        self.trend_threshold = 0.02  # 2% trend threshold
        self.rebalance_threshold = 0.1  # 10% deviation triggers rebalancing
        self.max_position_size = 0.5  # Maximum 50% of portfolio in a single asset
        
        # Trading state
        self.portfolio_weights = {}  # Target weights for each asset
        self.last_rebalance = datetime.now()
        self.market_regime = "neutral"  # Current market regime assessment
        self.macro_indicators = {
            "trend_strength": 0.0,
            "volatility": 0.0,
            "correlation": 0.0
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
        
        # Get current prices and portfolio value
        current_prices = {
            symbol: data['price'] 
            for symbol, data in market_data.items()
        }
        portfolio_value = self.get_portfolio_value(current_prices)
        available_usdt = self.wallet.get('USDT', 0)
        
        # Update market regime assessment
        self._update_market_regime(market_data)
        
        # Calculate target portfolio weights based on risk parity
        self._calculate_target_weights(market_data)
        
        for symbol, data in market_data.items():
            if 'price_change_percent_24h' not in data:
                continue
            
            base_symbol = symbol.replace('USDT', '')
            price = data['price']
            price_change_24h = data['price_change_percent_24h']
            
            # Calculate current position and target position
            holding_amount = self.wallet.get(base_symbol, 0)
            holding_value = holding_amount * price
            current_weight = holding_value / portfolio_value if portfolio_value > 0 else 0
            target_weight = self.portfolio_weights.get(symbol, 0.0)
            
            # Check if rebalancing is needed
            weight_diff = abs(current_weight - target_weight)
            if weight_diff > self.rebalance_threshold:
                if current_weight < target_weight and available_usdt > 0:
                    # Need to buy more
                    amount_to_buy = min(
                        available_usdt,
                        (target_weight - current_weight) * portfolio_value
                    )
                    if amount_to_buy >= 5.0:  # Minimum trade size
                        result['trades'].append({
                            'action': 'buy',
                            'symbol': symbol,
                            'amount_usdt': amount_to_buy,
                            'reason': f'Rebalancing: increasing {base_symbol} allocation'
                        })
                elif current_weight > target_weight and holding_amount > 0:
                    # Need to sell some
                    amount_to_sell = holding_amount * (
                        (current_weight - target_weight) / current_weight
                    )
                    if amount_to_sell * price >= 5.0:  # Minimum trade size
                        result['trades'].append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': amount_to_sell,
                            'reason': f'Rebalancing: reducing {base_symbol} allocation'
                        })
            
            result['analysis'][symbol] = {
                'price': price,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'market_regime': self.market_regime,
                'trend_strength': self.macro_indicators['trend_strength']
            }
        
        return result
    
    def _update_market_regime(self, market_data: Dict[str, Any]):
        """Update the current market regime assessment."""
        # Calculate average trend
        avg_change = sum(
            data.get('price_change_percent_24h', 0)
            for data in market_data.values()
        ) / len(market_data)
        
        # Calculate volatility
        volatility = sum(
            abs(data.get('price_change_percent_24h', 0))
            for data in market_data.values()
        ) / len(market_data)
        
        self.macro_indicators['trend_strength'] = avg_change
        self.macro_indicators['volatility'] = volatility
        
        # Update market regime
        if avg_change > self.trend_threshold and volatility < 5:
            self.market_regime = "bullish"
        elif avg_change < -self.trend_threshold and volatility < 5:
            self.market_regime = "bearish"
        elif volatility >= 5:
            self.market_regime = "volatile"
        else:
            self.market_regime = "neutral"
    
    def _calculate_target_weights(self, market_data: Dict[str, Any]):
        """Calculate target portfolio weights based on risk parity."""
        total_risk_score = 0
        risk_scores = {}
        
        for symbol, data in market_data.items():
            # Calculate risk score based on volatility and market regime
            volatility = abs(data.get('price_change_percent_24h', 0))
            base_risk = 1.0 / volatility if volatility > 0 else 1.0
            
            # Adjust risk based on market regime
            if self.market_regime == "bullish":
                risk_scores[symbol] = base_risk * 1.2
            elif self.market_regime == "bearish":
                risk_scores[symbol] = base_risk * 0.8
            else:
                risk_scores[symbol] = base_risk
            
            total_risk_score += risk_scores[symbol]
        
        # Calculate weights based on risk scores
        for symbol in market_data:
            self.portfolio_weights[symbol] = (
                risk_scores[symbol] / total_risk_score
                if total_risk_score > 0 else 0.0
            )
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List[Trader]) -> Optional[str]:
        """Generate a message about market insights."""
        if random.random() > 0.2:  # 20% chance to post a message
            return None
        
        messages = [
            f"Current market regime: {self.market_regime.upper()}. Adjusting risk allocations accordingly. 📊",
            f"Observing strong macro trends. Remember: cycles are the most fundamental influence. 🌊",
            "The key is to have a balanced, all-weather portfolio that can perform in any environment. ⚖️",
            "Risk parity is essential. Don't put all your eggs in one basket. 🥚",
            f"Market volatility at {self.macro_indicators['volatility']:.1f}%. Staying vigilant. 👀"
        ]
        
        return random.choice(messages)
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """Respond to other traders' messages."""
        if random.random() > 0.3:  # 30% chance to respond
            return None
        
        content = message['content'].lower()
        trader_name = message['trader_name']
        
        if "trend" in content or "market" in content:
            responses = [
                f"Interesting point, {trader_name}. The key is understanding the broader economic cycles. 🔄",
                f"I see it differently, {trader_name}. We need to focus on risk-adjusted returns across all weather conditions. ⚖️",
                f"Agree with {trader_name}. It's all about maintaining balance while markets evolve. 🎯"
            ]
            return random.choice(responses)
        
        return None 