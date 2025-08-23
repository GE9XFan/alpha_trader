-- SENTIMENT: news_sentiment
-- Generated from ACTUAL API response investigation
-- Total fields in response: 24

CREATE TABLE news_sentiment (
    id BIGSERIAL PRIMARY KEY,
    items TEXT,
    sentiment_score_definition TEXT,
    relevance_score_definition TEXT,
    title TEXT,
    url TEXT,
    time_published TEXT,
    authors TEXT,
    summary TEXT,
    banner_image TEXT,
    source TEXT,
    category_within_source TEXT,
    source_domain TEXT,
    topic TEXT,
    relevance_score TEXT,
    overall_sentiment_score NUMERIC,
    overall_sentiment_label TEXT,
    ticker TEXT,
    ticker_sentiment_score TEXT,
    ticker_sentiment_label TEXT    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

