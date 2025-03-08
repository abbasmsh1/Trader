import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from textblob import TextBlob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class MarketNewsFetcher:
    def __init__(self):
        """
        Initialize the MarketNewsFetcher with multiple API configurations.
        """
        # CryptoCompare configuration
        self.cryptocompare_url = "https://min-api.cryptocompare.com/data/v2"
        self.cryptocompare_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')
        
        # CoinGecko configuration (no API key required)
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
        # Messari configuration
        self.messari_url = "https://data.messari.io/api/v1"
        self.messari_key = os.getenv('MESSARI_API_KEY', '')
        
        # LunarCrush configuration
        self.lunarcrush_url = "https://lunarcrush.com/api/v2"
        self.lunarcrush_key = os.getenv('LUNARCRUSH_API_KEY', '')
        
        # Initialize coin ID mappings
        self.coin_ids = {}
        self._init_coin_mappings()
    
    def _init_coin_mappings(self):
        """Initialize coin ID mappings for different APIs."""
        try:
            # Get CoinGecko coin list
            response = requests.get(f"{self.coingecko_url}/coins/list")
            coins = response.json()
            for coin in coins:
                symbol = coin['symbol'].upper()
                if symbol not in self.coin_ids:
                    self.coin_ids[symbol] = {}
                self.coin_ids[symbol]['coingecko'] = coin['id']
        except Exception as e:
            print(f"Error initializing coin mappings: {str(e)}")
    
    def get_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Fetch news from multiple sources and combine them.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            limit (int): Maximum number of news articles to fetch per source
            
        Returns:
            List[Dict]: Combined list of news articles with sentiment analysis
        """
        clean_symbol = symbol.replace('/USDT', '').replace('USDT', '').upper()
        if not clean_symbol or clean_symbol == 'UNKNOWN':
            return []
        
        all_news = []
        
        # Use ThreadPoolExecutor to fetch news from multiple sources concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_source = {
                executor.submit(self._get_cryptocompare_news, clean_symbol, limit): 'cryptocompare',
                executor.submit(self._get_coingecko_news, clean_symbol, limit): 'coingecko',
                executor.submit(self._get_messari_news, clean_symbol, limit): 'messari',
                executor.submit(self._get_lunarcrush_news, clean_symbol, limit): 'lunarcrush'
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    news_items = future.result()
                    all_news.extend(news_items)
                except Exception as e:
                    print(f"Error fetching news from {source}: {str(e)}")
        
        # Sort by published date and limit total results
        all_news.sort(key=lambda x: x['published_at'], reverse=True)
        return all_news[:limit * 2]  # Return twice the limit since we're combining sources
    
    def _get_cryptocompare_news(self, symbol: str, limit: int) -> List[Dict]:
        """Fetch news from CryptoCompare."""
        try:
            headers = {'authorization': f'Apikey {self.cryptocompare_key}'}
            params = {
                'categories': symbol.lower(),
                'excludeCategories': 'Sponsored',
                'lang': 'EN',
                'limit': limit
            }
            
            response = requests.get(
                f"{self.cryptocompare_url}/news/",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            
            news_data = response.json()
            if not isinstance(news_data, dict) or news_data.get('Response') != 'Success':
                return []
            
            processed_news = []
            for article in news_data.get('Data', []):
                if not all(k in article for k in ['title', 'body', 'url', 'published_on']):
                    continue
                
                sentiment = self._analyze_sentiment(article['title'] + ' ' + article['body'])
                processed_news.append({
                    'title': article['title'],
                    'body': article['body'],
                    'url': article['url'],
                    'published_at': datetime.fromtimestamp(article['published_on']),
                    'source': f"CryptoCompare - {article.get('source', 'Unknown')}",
                    'sentiment_score': sentiment['score'],
                    'sentiment_magnitude': sentiment['magnitude'],
                    'sentiment_label': sentiment['label']
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching CryptoCompare news: {str(e)}")
            return []
    
    def _get_coingecko_news(self, symbol: str, limit: int) -> List[Dict]:
        """Fetch news from CoinGecko."""
        try:
            coin_id = self.coin_ids.get(symbol, {}).get('coingecko')
            if not coin_id:
                return []
            
            response = requests.get(
                f"{self.coingecko_url}/coins/{coin_id}/status_updates",
                params={'per_page': limit}
            )
            response.raise_for_status()
            
            updates = response.json().get('status_updates', [])
            processed_news = []
            
            for update in updates:
                if 'description' not in update:
                    continue
                
                sentiment = self._analyze_sentiment(update['description'])
                processed_news.append({
                    'title': update.get('category', 'Project Update'),
                    'body': update['description'],
                    'url': update.get('project_url', ''),
                    'published_at': datetime.strptime(update['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ'),
                    'source': 'CoinGecko',
                    'sentiment_score': sentiment['score'],
                    'sentiment_magnitude': sentiment['magnitude'],
                    'sentiment_label': sentiment['label']
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching CoinGecko news: {str(e)}")
            return []
    
    def _get_messari_news(self, symbol: str, limit: int) -> List[Dict]:
        """Fetch news from Messari."""
        try:
            headers = {'x-messari-api-key': self.messari_key}
            response = requests.get(
                f"{self.messari_url}/assets/{symbol.lower()}/news",
                headers=headers
            )
            response.raise_for_status()
            
            news_data = response.json().get('data', [])
            processed_news = []
            
            for article in news_data[:limit]:
                if not all(k in article for k in ['title', 'content', 'url', 'published_at']):
                    continue
                
                sentiment = self._analyze_sentiment(article['title'] + ' ' + article['content'])
                processed_news.append({
                    'title': article['title'],
                    'body': article['content'],
                    'url': article['url'],
                    'published_at': datetime.strptime(article['published_at'], '%Y-%m-%dT%H:%M:%SZ'),
                    'source': 'Messari',
                    'sentiment_score': sentiment['score'],
                    'sentiment_magnitude': sentiment['magnitude'],
                    'sentiment_label': sentiment['label']
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching Messari news: {str(e)}")
            return []
    
    def _get_lunarcrush_news(self, symbol: str, limit: int) -> List[Dict]:
        """Fetch news from LunarCrush."""
        try:
            params = {
                'data': 'feeds',
                'key': self.lunarcrush_key,
                'symbol': symbol,
                'limit': limit
            }
            
            response = requests.get(
                f"{self.lunarcrush_url}/feeds",
                params=params
            )
            response.raise_for_status()
            
            feeds = response.json().get('data', [])
            processed_news = []
            
            for feed in feeds:
                if not all(k in feed for k in ['title', 'description', 'url', 'time']):
                    continue
                
                sentiment = self._analyze_sentiment(feed['title'] + ' ' + feed['description'])
                processed_news.append({
                    'title': feed['title'],
                    'body': feed['description'],
                    'url': feed['url'],
                    'published_at': datetime.fromtimestamp(feed['time']),
                    'source': f"LunarCrush - {feed.get('source', 'Unknown')}",
                    'sentiment_score': sentiment['score'],
                    'sentiment_magnitude': sentiment['magnitude'],
                    'sentiment_label': sentiment['label']
                })
            
            return processed_news
        except Exception as e:
            print(f"Error fetching LunarCrush news: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze the sentiment of a text using TextBlob."""
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        magnitude = analysis.sentiment.subjectivity
        
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
        """Calculate overall market sentiment based on news from all sources."""
        news_articles = self.get_news(symbol, limit=20)
        
        if not news_articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'article_count': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'sources': {}
            }
        
        # Calculate overall sentiment metrics
        sentiment_scores = [article['sentiment_score'] for article in news_articles]
        sentiment_count = {
            'positive': len([s for s in sentiment_scores if s > 0.1]),
            'negative': len([s for s in sentiment_scores if s < -0.1]),
            'neutral': len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
        }
        
        # Calculate per-source metrics
        source_metrics = {}
        for article in news_articles:
            source = article['source'].split(' - ')[0]  # Get main source name
            if source not in source_metrics:
                source_metrics[source] = {
                    'count': 0,
                    'sentiment_sum': 0.0,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            
            metrics = source_metrics[source]
            metrics['count'] += 1
            metrics['sentiment_sum'] += article['sentiment_score']
            
            if article['sentiment_label'] == 'positive':
                metrics['positive'] += 1
            elif article['sentiment_label'] == 'negative':
                metrics['negative'] += 1
            else:
                metrics['neutral'] += 1
        
        # Calculate final metrics
        total_articles = len(news_articles)
        avg_sentiment = sum(sentiment_scores) / total_articles
        
        # Calculate confidence based on article count and sentiment consistency
        confidence = min(total_articles / 40.0, 1.0) * (1 - abs(avg_sentiment))
        
        return {
            'overall_sentiment': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
            'sentiment_score': avg_sentiment,
            'confidence': confidence,
            'article_count': total_articles,
            'positive_ratio': sentiment_count['positive'] / total_articles,
            'negative_ratio': sentiment_count['negative'] / total_articles,
            'neutral_ratio': sentiment_count['neutral'] / total_articles,
            'sources': {
                source: {
                    'article_count': metrics['count'],
                    'avg_sentiment': metrics['sentiment_sum'] / metrics['count'],
                    'sentiment_distribution': {
                        'positive': metrics['positive'] / metrics['count'],
                        'negative': metrics['negative'] / metrics['count'],
                        'neutral': metrics['neutral'] / metrics['count']
                    }
                } for source, metrics in source_metrics.items()
            }
        } 