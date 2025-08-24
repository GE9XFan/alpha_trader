# AlphaTrader Architecture

## Data Flow

```
Alpha Vantage (600/min)          IBKR
       |                           |
       ├── Options with Greeks     ├── Real-time quotes
       ├── Technical Indicators    ├── 5-second bars
       ├── Sentiment Analysis      └── Order Execution
       └── Historical Data (20yr)
                |                           |
                └──────────┬────────────────┘
                           |
                    Feature Engine
                           |
                      ML Predictor
                           |
                    Signal Generator
                           |
                     Risk Manager
                           |
                    Order Executor
```

## Key Design Decisions

1. **Greeks are PROVIDED**: Alpha Vantage provides all Greeks - no local calculation
2. **Dual Sources**: Best of both worlds - AV for analytics, IBKR for execution
3. **Cache First**: Multi-tier caching minimizes API calls
4. **Progressive Build**: Each component builds on previous work

## Component Responsibilities

| Component | Data Source | Responsibility |
|-----------|------------|----------------|
| AlphaVantageClient | Alpha Vantage | 38 APIs for options, indicators, sentiment |
| MarketDataManager | IBKR | Real-time quotes and execution |
| OptionsDataManager | Alpha Vantage | Options chains with Greeks |
| FeatureEngine | Both | 45 features from both sources |
| MLPredictor | Alpha Vantage | Trained on 20yr historical data |
| SignalGenerator | Both | Generate signals using all data |
| RiskManager | Alpha Vantage | Portfolio Greeks management |
| PaperTrader | Both | Simulated trading |
| LiveTrader | Both | Real money trading |
