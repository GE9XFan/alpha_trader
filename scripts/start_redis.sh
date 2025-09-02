#!/bin/bash
# Start Redis with AlphaTrader configuration

echo "Starting Redis for AlphaTrader Pro..."

# Check if Redis is already running
if pgrep -x "redis-server" > /dev/null
then
    echo "Redis is already running. Stopping existing instance..."
    redis-cli shutdown
    sleep 2
fi

# Start Redis with our custom configuration
redis-server /Users/michaelmerrick/AlphaTraderPro/config/redis.conf

# Wait for Redis to start
sleep 1

# Test connection
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Redis started successfully"
    echo "  Host: localhost"
    echo "  Port: 6379"
    echo "  Config: config/redis.conf"
    echo "  Data: data/redis/"
else
    echo "✗ Failed to start Redis"
    exit 1
fi