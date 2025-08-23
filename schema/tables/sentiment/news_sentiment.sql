-- News Sentiment
CREATE TABLE news_sentiment (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255),
    url VARCHAR(255),
    time_published VARCHAR(50),
    authors TEXT[],
    summary VARCHAR(50),
    banner_image VARCHAR(255),
    source VARCHAR(50),
    category_within_source VARCHAR(10),
    source_domain VARCHAR(50),
    topics TEXT[],
    overall_sentiment_score NUMERIC,
    overall_sentiment_label VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ticker sentiment (normalized)
CREATE TABLE {table_name}_tickers (
    id BIGSERIAL PRIMARY KEY,
    article_id VARCHAR(100),
    ticker VARCHAR(10) NOT NULL,
    relevance_score NUMERIC,
    ticker_sentiment_score NUMERIC,
    ticker_sentiment_label VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_{table_name}_published ON {table_name}(time_published) WHERE time_published IS NOT NULL;
CREATE INDEX idx_{table_name}_tickers_ticker ON {table_name}_tickers(ticker);
