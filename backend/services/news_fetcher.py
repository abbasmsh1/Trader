"""Crypto news aggregation service using free and open sources."""

import asyncio
import aiohttp
import feedparser
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class CryptoNewsFetcher:
    """Fetches and aggregates cryptocurrency news from various free sources."""
    
    def __init__(self, cache_dir: str = "data/news_cache"):
        """Initialize the news fetcher with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # RSS feeds from reputable open sources
        self.rss_sources = {
            'cointelegraph': 'https://cointelegraph.com/rss',
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'bitcoin_news': 'https://news.bitcoin.com/feed/',
            'crypto_daily': 'https://cryptodaily.co.uk/feed/',
        }
        
        # GitHub crypto project updates
        self.github_sources = [
            'bitcoin/bitcoin',
            'ethereum/go-ethereum',
            'binance-chain/bsc',
            'solana-labs/solana',
        ]
        
        # Cache settings
        self.cache_duration = timedelta(minutes=15)
    
    async def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all sources."""
        try:
            # Check cache first
            cached_news = self._load_cache()
            if cached_news:
                return cached_news
            
            # Fetch from all sources concurrently
            tasks = [
                self._fetch_rss_news(),
                self._fetch_github_updates(),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and sort news
            all_news = []
            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)
            
            # Sort by timestamp
            all_news.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Cache the results
            self._save_cache(all_news)
            
            return all_news
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    async def _fetch_rss_news(self) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds."""
        news_items = []
        
        for source, url in self.rss_sources.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)
                            
                            for entry in feed.entries[:10]:  # Get latest 10 entries
                                news_items.append({
                                    'source': source,
                                    'title': entry.title,
                                    'summary': entry.summary,
                                    'url': entry.link,
                                    'timestamp': datetime.strptime(
                                        entry.published, 
                                        '%a, %d %b %Y %H:%M:%S %z'
                                    ).isoformat(),
                                    'type': 'news'
                                })
            except Exception as e:
                logger.error(f"Error fetching RSS from {source}: {e}")
        
        return news_items
    
    async def _fetch_github_updates(self) -> List[Dict[str, Any]]:
        """Fetch latest updates from major crypto projects on GitHub."""
        news_items = []
        
        async with aiohttp.ClientSession() as session:
            for repo in self.github_sources:
                try:
                    # Fetch latest releases
                    url = f"https://api.github.com/repos/{repo}/releases"
                    async with session.get(url) as response:
                        if response.status == 200:
                            releases = await response.json()
                            
                            for release in releases[:3]:  # Get latest 3 releases
                                news_items.append({
                                    'source': 'github',
                                    'project': repo,
                                    'title': f"New {repo} release: {release['tag_name']}",
                                    'summary': release['body'][:500],  # Truncate long descriptions
                                    'url': release['html_url'],
                                    'timestamp': release['published_at'],
                                    'type': 'development'
                                })
                except Exception as e:
                    logger.error(f"Error fetching GitHub updates for {repo}: {e}")
        
        return news_items
    
    def _get_cache_file(self) -> Path:
        """Get the cache file path."""
        return self.cache_dir / "news_cache.json"
    
    def _load_cache(self) -> List[Dict[str, Any]]:
        """Load news from cache if still valid."""
        cache_file = self._get_cache_file()
        
        if not cache_file.exists():
            return []
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cache_time < self.cache_duration:
                return cached_data['news']
        except Exception as e:
            logger.error(f"Error loading news cache: {e}")
        
        return []
    
    def _save_cache(self, news: List[Dict[str, Any]]) -> None:
        """Save news to cache."""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'news': news
            }
            
            with open(self._get_cache_file(), 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving news cache: {e}")
    
    def analyze_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Basic sentiment analysis of news items.
        Uses a simple keyword-based approach since we're avoiding paid APIs.
        """
        positive_keywords = {'surge', 'gain', 'bull', 'up', 'high', 'positive', 'growth'}
        negative_keywords = {'crash', 'drop', 'bear', 'down', 'low', 'negative', 'loss'}
        
        sentiment_scores = {
            'overall': 0,
            'by_source': {},
            'by_type': {},
            'trending_topics': {}
        }
        
        for item in news_items:
            # Calculate basic sentiment score
            text = f"{item['title']} {item['summary']}".lower()
            score = sum(1 for word in positive_keywords if word in text)
            score -= sum(1 for word in negative_keywords if word in text)
            
            # Update overall sentiment
            sentiment_scores['overall'] += score
            
            # Update source sentiment
            source = item['source']
            if source not in sentiment_scores['by_source']:
                sentiment_scores['by_source'][source] = []
            sentiment_scores['by_source'][source].append(score)
            
            # Update type sentiment
            news_type = item.get('type', 'news')
            if news_type not in sentiment_scores['by_type']:
                sentiment_scores['by_type'][news_type] = []
            sentiment_scores['by_type'][news_type].append(score)
        
        # Average out the scores
        if news_items:
            sentiment_scores['overall'] /= len(news_items)
            
            for source in sentiment_scores['by_source']:
                scores = sentiment_scores['by_source'][source]
                sentiment_scores['by_source'][source] = sum(scores) / len(scores)
            
            for news_type in sentiment_scores['by_type']:
                scores = sentiment_scores['by_type'][news_type]
                sentiment_scores['by_type'][news_type] = sum(scores) / len(scores)
        
        return sentiment_scores
