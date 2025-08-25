#!/usr/bin/env python3
"""
Test NEWS_SENTIMENT API directly to debug why no articles are returned
"""
import aiohttp
import asyncio
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('AV_API_KEY')
BASE_URL = "https://www.alphavantage.co/query"

async def test_news_sentiment(tickers=None, **kwargs):
    """Test NEWS_SENTIMENT with various parameters"""
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': API_KEY
    }
    
    # Add optional parameters
    if tickers:
        params['tickers'] = tickers
    
    for key, value in kwargs.items():
        if value:
            params[key] = value
    
    print(f"\n📰 Testing NEWS_SENTIMENT")
    print(f"Parameters: {params}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(BASE_URL, params=params) as resp:
            data = await resp.json()
            
            # Show response structure
            print(f"Response keys: {list(data.keys())}")
            
            # Check for Information message (error)
            if 'Information' in data:
                print(f"⚠️ API Information: {data['Information']}")
            
            # Check for feed
            if 'feed' in data:
                feed = data['feed']
                print(f"Feed contains {len(feed)} articles")
                
                if len(feed) > 0:
                    # Show first article
                    article = feed[0]
                    print(f"\nFirst article keys: {list(article.keys())}")
                    print(f"Title: {article.get('title', 'N/A')[:100]}")
                    print(f"Source: {article.get('source', 'N/A')}")
                    print(f"Time: {article.get('time_published', 'N/A')}")
                    
                    # Check ticker sentiment
                    if 'ticker_sentiment' in article:
                        print(f"Ticker sentiment: {article['ticker_sentiment'][:2]}")
            
            # Check other metadata
            if 'items' in data:
                print(f"Items: {data['items']}")
            
            if 'sentiment_score_definition' in data:
                print(f"Sentiment definitions provided: Yes")
            
            if 'relevance_score_definition' in data:
                print(f"Relevance definitions provided: Yes")
            
            return data

async def main():
    print("="*70)
    print("TESTING ALPHA VANTAGE NEWS_SENTIMENT API")
    print("="*70)
    print(f"API Key: {API_KEY[:8]}...")
    
    # Test 1a: With multiple tickers (no spaces)
    print("\n" + "="*50)
    print("TEST 1a: Multiple Tickers - No Spaces (SPY,AAPL)")
    print("="*50)
    result = await test_news_sentiment(tickers='SPY,AAPL', limit=50)
    
    # Test 1b: With multiple tickers (with spaces)
    print("\n" + "="*50)
    print("TEST 1b: Multiple Tickers - With Spaces (SPY, AAPL)")
    print("="*50)
    result = await test_news_sentiment(tickers='SPY, AAPL', limit=50)
    
    # Test 1c: With single ticker
    print("\n" + "="*50)
    print("TEST 1c: Single Ticker (SPY)")
    print("="*50)
    result = await test_news_sentiment(tickers='SPY', limit=50)
    
    # Test 2: Without tickers (general market news)
    print("\n" + "="*50)
    print("TEST 2: Without Tickers (General Market)")
    print("="*50)
    result = await test_news_sentiment(limit=50)
    
    # Test 3: With topics
    print("\n" + "="*50)
    print("TEST 3: With Topics (technology)")
    print("="*50)
    result = await test_news_sentiment(topics='technology', limit=50)
    
    # Test 4: With time range (last 24 hours)
    print("\n" + "="*50)
    print("TEST 4: With Time Range (Last 24 Hours)")
    print("="*50)
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    time_from = yesterday.strftime('%Y%m%dT%H%M')
    time_to = now.strftime('%Y%m%dT%H%M')
    result = await test_news_sentiment(
        tickers='AAPL',
        time_from=time_from,
        time_to=time_to,
        limit=50
    )
    
    # Test 5: Different sort order
    print("\n" + "="*50)
    print("TEST 5: With RELEVANCE Sort")
    print("="*50)
    result = await test_news_sentiment(
        tickers='TSLA',
        sort='RELEVANCE',
        limit=10
    )
    
    # Check if it's a weekend issue
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    weekday = datetime.now().strftime('%A')
    print(f"Today is: {weekday}")
    if weekday in ['Saturday', 'Sunday']:
        print("⚠️ It's a weekend - news volume may be lower")
    
    market_hour = datetime.now().hour
    if market_hour < 9 or market_hour > 16:
        print(f"⚠️ Outside market hours (current hour: {market_hour})")
    
    print("\n💡 Possible reasons for no articles:")
    print("1. API rate limit or quota issue")
    print("2. Weekend/after-hours - less news")
    print("3. Ticker parameter filtering too restrictive")
    print("4. Time range too narrow")
    print("5. API subscription tier limitation")

if __name__ == "__main__":
    asyncio.run(main())