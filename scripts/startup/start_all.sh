#!/bin/bash
# Start all AlphaTrader components

echo "🚀 Starting AlphaTrader System..."
echo "📊 Data Sources: Alpha Vantage (options/Greeks) + IBKR (execution)"

# Check environment
if [ -z "$AV_API_KEY" ]; then
    echo "❌ Error: AV_API_KEY not set"
    exit 1
fi

# Start components
python scripts/startup/start_market_data.py &
python scripts/startup/start_paper_trader.py &
python scripts/startup/start_discord_bot.py &

echo "✅ All components started"
