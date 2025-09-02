#!/bin/bash
# Start AlphaTrader Pro System

echo "========================================="
echo "    Starting AlphaTrader Pro System"
echo "========================================="

# Start Redis first
echo "[1/2] Starting Redis..."
./scripts/start_redis.sh

if [ $? -ne 0 ]; then
    echo "Failed to start Redis. Exiting."
    exit 1
fi

# Start main application
echo "[2/2] Starting AlphaTrader..."
python main.py