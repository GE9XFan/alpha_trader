#!/bin/bash
# Stop AlphaTrader Pro System

echo "========================================="
echo "    Stopping AlphaTrader Pro System"
echo "========================================="

# Kill Python processes
echo "Stopping AlphaTrader processes..."
pkill -f "python main.py"

# Stop Redis
echo "Stopping Redis..."
redis-cli shutdown

echo "✓ System stopped"