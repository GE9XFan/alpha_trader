#!/usr/bin/env python3
"""
Sentiment Processor - Alpha Vantage Sentiment and Technical Analysis
Part of AlphaTrader Pro System

This module operates independently and communicates only via Redis.
Redis keys used:
- sentiment:{symbol}: Sentiment analysis data
- technicals:{symbol}:{indicator}: Technical indicator data
"""

import asyncio
import json
import redis.asyncio as aioredis
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging
from redis_keys import Keys


class SentimentProcessor:
    """
    Processes Alpha Vantage sentiment and technical indicator data.
    """

    def __init__(self, config: Dict[str, Any], redis_conn: aioredis.Redis):
        """Initialize sentiment processor."""
        self.config = config
        self.redis = redis_conn
        self.logger = logging.getLogger(__name__)
        av_config = config.get('alpha_vantage', {})
        sentiment_config = av_config.get('sentiment', {})
        default_skip = ['SPY', 'QQQ', 'IWM', 'VXX']
        self.skip_sentiment_symbols = set(sentiment_config.get('skip_symbols', default_skip))
        self.ttls = config['modules']['data_ingestion']['store_ttls']

    @staticmethod
    def _parse_time(value: str) -> Optional[datetime]:
        if not value:
            return None

        formats = ('%Y-%m-%dT%H:%M:%SZ', '%Y%m%dT%H%M%S')
        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue
        return None

    async def fetch_sentiment(self, session: aiohttp.ClientSession, symbol: str,
                              rate_limiter, base_url: str, api_key: str, ttls: Dict):
        """Fetch sentiment data."""
        # Skip sentiment for configured symbols (primarily ETFs)
        if symbol in self.skip_sentiment_symbols:
            self.logger.info(f"Skipping sentiment for ETF {symbol} (not supported by Alpha Vantage)")
            return None

        try:
            await rate_limiter.acquire()

            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': api_key,
                'sort': 'LATEST',
                'limit': 50
            }

            async with session.get(base_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Check for API errors
                    if 'Error Message' in data:
                        self.logger.error(f"AV sentiment error for {symbol}: {data['Error Message']}")
                        return None

                    # Check for rate limits
                    if 'Note' in data:
                        self.logger.warning(f"AV sentiment soft limit: {data['Note']}")
                        await asyncio.sleep(30)
                        return None

                    if 'Information' in data:
                        self.logger.warning(f"AV sentiment info: {data['Information']}")
                        await asyncio.sleep(60)
                        return None

                    if 'feed' in data:
                        await self._store_sentiment_data(symbol, data, ttls)
                        self.logger.info(f"✓ Fetched sentiment for {symbol}: {len(data.get('feed', []))} articles")
                    else:
                        self.logger.warning(f"AV sentiment missing 'feed' for {symbol}; keys={list(data.keys())[:6]}")
                        return None

                    return data

        except Exception as e:
            self.logger.error(f"Sentiment fetch error for {symbol}: {e}")
            return None

    async def fetch_technicals(self, session: aiohttp.ClientSession, symbol: str,
                                rate_limiter, base_url: str, api_key: str, ttls: Dict):
        """Fetch all technical indicators: RSI, MACD, BBANDS, ATR."""
        try:
            # Define all indicators to fetch
            indicators = [
                {
                    'function': 'RSI',
                    'params': {
                        'function': 'RSI',
                        'symbol': symbol,
                        'interval': '5min',
                        'time_period': 14,
                        'series_type': 'close',
                        'apikey': api_key
                    },
                    'key': 'Technical Analysis: RSI',
                    'value_key': 'RSI'
                },
                {
                    'function': 'MACD',
                    'params': {
                        'function': 'MACD',
                        'symbol': symbol,
                        'interval': '5min',
                        'series_type': 'close',
                        'apikey': api_key
                    },
                    'key': 'Technical Analysis: MACD',
                    'value_keys': ['MACD', 'MACD_Signal', 'MACD_Hist']
                },
                {
                    'function': 'BBANDS',
                    'params': {
                        'function': 'BBANDS',
                        'symbol': symbol,
                        'interval': '5min',
                        'time_period': 20,
                        'series_type': 'close',
                        'nbdevup': 2,
                        'nbdevdn': 2,
                        'apikey': api_key
                    },
                    'key': 'Technical Analysis: BBANDS',
                    'value_keys': ['Real Upper Band', 'Real Middle Band', 'Real Lower Band']
                },
                {
                    'function': 'ATR',
                    'params': {
                        'function': 'ATR',
                        'symbol': symbol,
                        'interval': '5min',
                        'time_period': 14,
                        'apikey': api_key
                    },
                    'key': 'Technical Analysis: ATR',
                    'value_key': 'ATR'
                }
            ]

            # Fetch all indicators in parallel
            tasks = []
            for indicator in indicators:
                tasks.append(self._fetch_single_indicator(session, symbol, indicator, rate_limiter, base_url, ttls))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful fetches
            success_count = sum(1 for r in results if r and not isinstance(r, Exception))

            if success_count > 0:
                self.logger.info(f"✓ Fetched {success_count}/{len(indicators)} technicals for {symbol}")
                return {'success': success_count, 'total': len(indicators)}
            else:
                self.logger.warning(f"Failed to fetch any technicals for {symbol}")
                return None

        except Exception as e:
            self.logger.error(f"Technicals fetch error for {symbol}: {e}")
            return None

    async def _fetch_single_indicator(self, session: aiohttp.ClientSession, symbol: str,
                                       indicator: Dict, rate_limiter, base_url: str, ttls: Dict):
        """Fetch a single technical indicator."""
        try:
            await rate_limiter.acquire()

            async with session.get(base_url, params=indicator['params']) as response:
                response.raise_for_status()
                data = await response.json()

                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f"AV {indicator['function']} error for {symbol}: {data['Error Message']}")
                    return None

                # Check for rate limits
                if 'Note' in data:
                    self.logger.warning(f"AV {indicator['function']} soft limit: {data['Note']}")
                    await asyncio.sleep(30)
                    return None

                if 'Information' in data:
                    self.logger.warning(f"AV {indicator['function']} info: {data['Information']}")
                    await asyncio.sleep(60)
                    return None

                # Process and store the indicator data
                if indicator['key'] in data:
                    await self._store_technical_data(symbol, indicator['function'], data, indicator, ttls)
                    return data
                else:
                    self.logger.warning(f"AV {indicator['function']} missing '{indicator['key']}' for {symbol}")
                    return None

        except Exception as e:
            self.logger.error(f"Error fetching {indicator['function']} for {symbol}: {e}")
            return None

    async def _store_sentiment_data(self, symbol: str, data: Dict, ttls: Dict):
        """Store complete sentiment data to Redis including all article details."""
        try:
            feed = data.get('feed', [])

            if feed:
                # Process and store complete feed data
                articles = []
                ticker_sentiments = []
                source_counts: Dict[str, int] = {}
                article_ages: List[float] = []

                for article in feed:
                    # Store complete article data
                    article_data = {
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'time_published': article.get('time_published', ''),
                        'authors': article.get('authors', []),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'topics': article.get('topics', []),  # Includes topic and relevance_score
                        'overall_sentiment_score': article.get('overall_sentiment_score', 0),
                        'overall_sentiment_label': article.get('overall_sentiment_label', 'Neutral'),
                        'ticker_sentiment': []
                    }

                    source = article_data.get('source')
                    if source:
                        source_counts[source] = source_counts.get(source, 0) + 1

                    published_dt = self._parse_time(article_data.get('time_published', ''))
                    if published_dt:
                        age_minutes = max(0.0, (datetime.now(timezone.utc) - published_dt).total_seconds() / 60.0)
                        article_ages.append(age_minutes)

                    # Process ticker-specific sentiment for this article
                    if 'ticker_sentiment' in article:
                        for ticker_data in article['ticker_sentiment']:
                            if ticker_data.get('ticker') == symbol:
                                ticker_sent = {
                                    'ticker': ticker_data.get('ticker'),
                                    'relevance_score': float(ticker_data.get('relevance_score', 0)),
                                    'ticker_sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                                    'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label', 'Neutral')
                                }
                                article_data['ticker_sentiment'].append(ticker_sent)
                                ticker_sentiments.append(ticker_sent)

                    articles.append(article_data)

                # Calculate aggregate metrics
                avg_sentiment = 0
                avg_relevance = 0
                sentiment_distribution = {'Bullish': 0, 'Somewhat-Bullish': 0, 'Neutral': 0, 'Somewhat-Bearish': 0, 'Bearish': 0}

                if ticker_sentiments:
                    sentiment_scores = [ts['ticker_sentiment_score'] for ts in ticker_sentiments]
                    relevance_scores = [ts['relevance_score'] for ts in ticker_sentiments]
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    avg_relevance = sum(relevance_scores) / len(relevance_scores)

                    # Count sentiment distribution
                    for ts in ticker_sentiments:
                        label = ts.get('ticker_sentiment_label', 'Neutral')
                        if label in sentiment_distribution:
                            sentiment_distribution[label] += 1

                # Store complete sentiment data
                sentiment_data = {
                    'symbol': symbol,
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'article_count': len(feed),
                    'articles': articles,  # Complete article data
                    'aggregate': {
                        'avg_sentiment_score': avg_sentiment,
                        'avg_relevance_score': avg_relevance,
                        'sentiment_distribution': sentiment_distribution,
                        'total_mentions': len(ticker_sentiments)
                    },
                    'sentiment_definitions': data.get('sentiment_score_definition', ''),
                    'relevance_definition': data.get('relevance_score_definition', '')
                }

                await self.redis.setex(
                    f'sentiment:{symbol}',
                    ttls['sentiment'],
                    json.dumps(sentiment_data)
                )

                avg_age_minutes = None
                fresh_pct = None
                if article_ages:
                    avg_age_minutes = sum(article_ages) / len(article_ages)
                    fresh_count = sum(1 for age in article_ages if age <= 60)
                    fresh_pct = round((fresh_count / len(article_ages)) * 100, 2)

                monitoring_payload = {
                    'symbol': symbol,
                    'timestamp': sentiment_data['timestamp'],
                    'article_count': len(feed),
                    'avg_sentiment': avg_sentiment,
                    'avg_relevance': avg_relevance,
                    'avg_age_minutes': round(avg_age_minutes, 2) if avg_age_minutes is not None else None,
                    'fresh_articles_pct': fresh_pct,
                    'sources': source_counts,
                    'total_mentions': len(ticker_sentiments)
                }

                await self.redis.setex(
                    f'monitoring:sentiment:{symbol}',
                    self.ttls.get('monitoring', 60),
                    json.dumps(monitoring_payload)
                )

                # Determine overall label based on average sentiment
                overall_label = 'Neutral'
                if avg_sentiment <= -0.35:
                    overall_label = 'Bearish'
                elif -0.35 < avg_sentiment <= -0.15:
                    overall_label = 'Somewhat-Bearish'
                elif 0.15 <= avg_sentiment < 0.35:
                    overall_label = 'Somewhat-Bullish'
                elif avg_sentiment >= 0.35:
                    overall_label = 'Bullish'

                self.logger.info(
                    f"Stored sentiment for {symbol}: score={avg_sentiment:.3f} ({overall_label}), "
                    f"relevance={avg_relevance:.3f}, articles={len(feed)}, mentions={len(ticker_sentiments)}"
                )

        except Exception as e:
            self.logger.error(f"Error storing sentiment for {symbol}: {e}")

    async def _store_technical_data(self, symbol: str, indicator_name: str, data: Dict,
                                     indicator_config: Dict, ttls: Dict):
        """Store technical indicator data to Redis."""
        try:
            tech_key = f'Technical Analysis: {indicator_name}'
            if tech_key in data:
                values = data[tech_key]

                # Get latest date
                latest_date = sorted(values.keys())[0]
                latest_values = values[latest_date]

                metadata = {
                    'interval': indicator_config['params'].get('interval'),
                    'lookback': indicator_config['params'].get('time_period'),
                    'source': 'alpha_vantage'
                }

                # Handle multi-value indicators (MACD, BBANDS)
                if 'value_keys' in indicator_config:
                    # Multi-value indicator like MACD or BBANDS
                    tech_data = {
                        'symbol': symbol,
                        'indicator': indicator_name,
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'metadata': metadata
                    }

                    # Add each value component
                    for value_key in indicator_config['value_keys']:
                        if value_key in latest_values:
                            # Store with cleaner keys
                            clean_key = value_key.replace(' ', '_').replace('Real_', '').lower()
                            tech_data[clean_key] = float(latest_values[value_key])

                    await self.redis.setex(
                        f'technicals:{symbol}:{indicator_name.lower()}',
                        ttls['technicals'],
                        json.dumps(tech_data)
                    )

                    # Log based on indicator type
                    if indicator_name == 'MACD':
                        self.logger.info(f"Stored MACD for {symbol}: MACD={tech_data.get('macd', 0):.4f}, Signal={tech_data.get('macd_signal', 0):.4f}, Hist={tech_data.get('macd_hist', 0):.4f}")
                    elif indicator_name == 'BBANDS':
                        self.logger.info(f"Stored BBANDS for {symbol}: Upper={tech_data.get('upper_band', 0):.2f}, Middle={tech_data.get('middle_band', 0):.2f}, Lower={tech_data.get('lower_band', 0):.2f}")
                    else:
                        self.logger.info(f"Stored {indicator_name} for {symbol}: {tech_data}")

                else:
                    # Single-value indicator like RSI or ATR
                    value_key = indicator_config.get('value_key', indicator_name)
                    latest_value = float(latest_values[value_key])

                    tech_data = {
                        'symbol': symbol,
                        'indicator': indicator_name,
                        'value': latest_value,
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'metadata': metadata
                    }

                    await self.redis.setex(
                        f'technicals:{symbol}:{indicator_name.lower()}',
                        ttls['technicals'],
                        json.dumps(tech_data)
                    )

                    self.logger.info(f"Stored {indicator_name} for {symbol}: {latest_value:.2f}")

        except Exception as e:
            self.logger.error(f"Error storing {indicator_name} for {symbol}: {e}")