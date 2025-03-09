"""News-focused trading agent that makes decisions based on market sentiment."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models.trader import Trader

logger = logging.getLogger(__name__)

class NewsTrader(Trader):
    """Trading agent that specializes in news-based trading decisions."""
    
    def __init__(self):
        super().__init__(
            name="Nina NewsBot",
            personality="Analytical and news-driven",
            trading_style="News sentiment trading with technical validation",
            backstory=(
                "A former financial journalist turned algorithmic trader. "
                "Specializes in analyzing market sentiment and news impact "
                "on cryptocurrency prices."
            )
        )
        
        # News sensitivity settings
        self.sentiment_threshold = 0.3  # Minimum sentiment score to trigger action
        self.max_position_size = 0.4    # Maximum portion of portfolio to risk
        self.min_confidence = 0.6       # Minimum confidence for trade execution
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and news sentiment to make trading decisions.
        
        Args:
            market_data: Current market data
            
        Returns:
            Analysis results including potential trades
        """
        # Get base analysis from parent class
        analysis = await super().analyze_market(market_data)
        
        if not self.active:
            return analysis
        
        try:
            portfolio_value = self.get_portfolio_value(market_data)
            trades = []
            
            # Analyze each trading pair
            for symbol, data in market_data.items():
                base_symbol = symbol.replace('USDT', '')
                
                # Get news sentiment impact
                sentiment_score = self._evaluate_news_impact(base_symbol)
                
                # Skip if sentiment is not strong enough
                if abs(sentiment_score) < self.sentiment_threshold:
                    continue
                
                # Calculate position size based on sentiment strength
                position_size = min(
                    abs(sentiment_score) * self.max_position_size * portfolio_value,
                    self.wallet.get('USDT', 0) * 0.95  # Leave some for fees
                )
                
                if sentiment_score > 0:  # Bullish sentiment
                    current_holding = self.wallet.get(base_symbol, 0)
                    
                    # If we don't have a position and sentiment is strong, buy
                    if current_holding == 0 and position_size >= 10:  # Minimum trade size
                        trades.append({
                            'action': 'buy',
                            'symbol': symbol,
                            'amount_usdt': position_size,
                            'reason': f"Strong positive sentiment ({sentiment_score:.2f})"
                        })
                
                elif sentiment_score < 0:  # Bearish sentiment
                    current_holding = self.wallet.get(base_symbol, 0)
                    
                    # If we have a position and sentiment turned negative, sell
                    if current_holding > 0:
                        # Calculate sell amount based on sentiment strength
                        sell_portion = abs(sentiment_score)
                        amount_to_sell = current_holding * sell_portion
                        
                        trades.append({
                            'action': 'sell',
                            'symbol': symbol,
                            'amount_crypto': amount_to_sell,
                            'reason': f"Strong negative sentiment ({sentiment_score:.2f})"
                        })
            
            analysis['trades'] = trades
            return analysis
            
        except Exception as e:
            logger.error(f"Error in news trader analysis: {e}")
            return analysis
    
    def generate_message(self, market_data: Dict[str, Any], other_traders: List['Trader']) -> Optional[str]:
        """Generate insights based on news analysis."""
        if not self.active or not self.news_sentiment:
            return None
        
        try:
            # Get overall market sentiment
            overall_sentiment = self.news_sentiment.get('overall', 0)
            
            # Get development activity sentiment
            dev_sentiment = self.news_sentiment.get('by_type', {}).get('development', 0)
            
            # Construct message
            message_parts = []
            
            # Add sentiment analysis
            if abs(overall_sentiment) > 0.3:
                sentiment_str = "bullish" if overall_sentiment > 0 else "bearish"
                message_parts.append(f"Market sentiment is strongly {sentiment_str}")
            
            # Add development insights
            if abs(dev_sentiment) > 0.5:
                dev_str = "positive" if dev_sentiment > 0 else "concerning"
                message_parts.append(f"Development activity is showing {dev_str} signals")
            
            # Add source reliability
            reliable_sources = [
                source for source, score in self.news_sentiment.get('by_source', {}).items()
                if abs(score) > 0.7
            ]
            if reliable_sources:
                sources_str = ", ".join(reliable_sources)
                message_parts.append(f"Strong signals from {sources_str}")
            
            if message_parts:
                return f"{self.name}'s Analysis: {'. '.join(message_parts)}."
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating news trader message: {e}")
            return None
    
    def respond_to_message(self, message: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[str]:
        """Respond to other traders' messages with news-based insights."""
        if not self.active or not self.news_sentiment:
            return None
        
        try:
            # Only respond to messages that might benefit from news insight
            content = message['content'].lower()
            if any(word in content for word in ['news', 'sentiment', 'market', 'development', 'update']):
                # Get relevant sentiment scores
                overall_sentiment = self.news_sentiment.get('overall', 0)
                dev_sentiment = self.news_sentiment.get('by_type', {}).get('development', 0)
                
                # Construct response based on message content
                if 'development' in content or 'github' in content:
                    if abs(dev_sentiment) > 0.3:
                        activity = "positive" if dev_sentiment > 0 else "negative"
                        return f"Recent development activity is showing {activity} trends."
                
                elif 'news' in content or 'sentiment' in content:
                    if abs(overall_sentiment) > 0.3:
                        outlook = "optimistic" if overall_sentiment > 0 else "cautious"
                        return f"Based on recent news, I'm feeling {outlook} about the market."
            
            return None
            
        except Exception as e:
            logger.error(f"Error in news trader response: {e}")
            return None
