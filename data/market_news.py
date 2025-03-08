import requests
from typing import List, Dict
from datetime import datetime, timedelta
import os
from textblob import TextBlob
import json

class MarketNewsFetcher:
    def __init__(self):
        """
        Initialize the MarketNewsFetcher with API configuration.
        Uses CryptoCompare News API for real-time crypto news.
        """
        self.base_url = "https://min-api.cryptocompare.com/data/v2"
        # Note: Replace with your actual API key
        self.api_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')
        self.headers = {'authorization': f'Apikey {self.api_key}'}
        
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Fetch recent news articles related to a specific cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            limit (int): Maximum number of news articles to fetch
            
        Returns:
            List[Dict]: List of news articles with sentiment analysis
        """
        try:
            # Remove /USDT if present
            clean_symbol = symbol.replace('/USDT', '')
            
            # Fetch news from CryptoCompare
            endpoint = f"{self.base_url}/news/"
            params = {
                'categories': clean_symbol.lower(),
                'excludeCategories': 'Sponsored',
                'lang': 'EN',
                'limit': limit
            }
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            if news_data['Response'] != 'Success':
                return []
            
            # Process and analyze each news article
            processed_news = []
            for article in news_data['Data']:
                # Perform sentiment analysis
                sentiment = self._analyze_sentiment(article['title'] + ' ' + article['body'])
                
                processed_news.append({
                    'title': article['title'],
                    'body': article['body'],
                    'url': article['url'],
                    'published_at': article['published_on'],
                    'source': article['source'],
                    'sentiment_score': sentiment['score'],
                    'sentiment_magnitude': sentiment['magnitude'],
                    'sentiment_label': sentiment['label']
                })
            
            return processed_news
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze the sentiment of a text using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        analysis = TextBlob(text)
        
        # Get polarity score (-1 to 1) and subjectivity (0 to 1)
        score = analysis.sentiment.polarity
        magnitude = analysis.sentiment.subjectivity
        
        # Determine sentiment label
        if score > 0.1:
            label = 'positive'
        elif score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': score,
            'magnitude': magnitude,
            'label': label
        }
    
    def get_market_sentiment(self, symbol: str, timeframe: str = '24h') -> Dict:
        """
        Calculate overall market sentiment based on recent news.
        
        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Time window for news analysis
            
        Returns:
            Dict: Aggregated sentiment metrics
        """
        news_articles = self.get_news(symbol, limit=20)
        
        if not news_articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'article_count': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        # Calculate sentiment metrics
        sentiment_scores = [article['sentiment_score'] for article in news_articles]
        sentiment_count = {
            'positive': len([s for s in sentiment_scores if s > 0.1]),
            'negative': len([s for s in sentiment_scores if s < -0.1]),
            'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
        }
        
        total_articles = len(news_articles)
        avg_sentiment = sum(sentiment_scores) / total_articles
        
        # Calculate confidence based on article count and sentiment consistency
        confidence = min(total_articles / 20.0, 1.0) * (1 - abs(avg_sentiment))
        
        return {
            'overall_sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
            'sentiment_score': avg_sentiment,
            'confidence': confidence,
            'article_count': total_articles,
            'positive_ratio': sentiment_count['positive'] / total_articles,
            'negative_ratio': sentiment_count['negative'] / total_articles,
            'neutral_ratio': sentiment_count['neutral'] / total_articles
        } 