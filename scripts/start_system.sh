#!/bin/bash

# AlphaTrader System Startup Script
# Production-grade startup with health checks and validation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "AlphaTrader System Startup"
echo "=========================================="
echo "Time: $(date)"
echo "Mode: ${TRADING_MODE:-development}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python version
echo -n "Checking Python version... "
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
REQUIRED_VERSION="3.11"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION (requires >= $REQUIRED_VERSION)"
    exit 1
fi

# Check for .env file
echo -n "Checking environment configuration... "
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file found"
    # Load environment variables
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}⚠${NC} .env file not found"
    echo "  Creating from template..."
    cp .env.template .env
    echo -e "${RED}Please configure .env file with your API keys and settings${NC}"
    exit 1
fi

# Validate required environment variables
echo "Validating configuration..."
REQUIRED_VARS=("IBKR_ACCOUNT" "ALPHA_VANTAGE_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
        echo -e "  ${RED}✗${NC} $var is not set"
    else
        echo -e "  ${GREEN}✓${NC} $var is configured"
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}Missing required environment variables. Please configure .env file.${NC}"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p logs/dev
mkdir -p logs/prod
mkdir -p logs/test
echo -e "  ${GREEN}✓${NC} Log directories created"

# Run health check
echo ""
echo "Running system health check..."
python3 scripts/health_check.py
HEALTH_CHECK_RESULT=$?

if [ $HEALTH_CHECK_RESULT -ne 0 ]; then
    echo -e "${RED}Health check failed. Please fix issues before starting.${NC}"
    exit 1
fi

# Determine startup mode
MODE=${TRADING_MODE:-development}
CONFIG_FILE="config/${MODE}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}System Ready to Start${NC}"
echo "=========================================="
echo "Configuration: $CONFIG_FILE"
echo "Log directory: logs/$MODE"
echo ""

# Start the main application
echo "Starting AlphaTrader in $MODE mode..."
echo ""

# Check if running in development mode
if [ "$MODE" = "development" ]; then
    echo -e "${YELLOW}Running in DEVELOPMENT mode - Paper trading only${NC}"
    echo ""
fi

# Check if running in production mode
if [ "$MODE" = "production" ]; then
    echo -e "${RED}WARNING: Running in PRODUCTION mode - Real money trading${NC}"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Startup cancelled"
        exit 0
    fi
fi

# Launch the application
echo "Launching AlphaTrader..."
python3 -m src.core.main --config "$CONFIG_FILE" --mode "$MODE"

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}AlphaTrader shutdown cleanly${NC}"
else
    echo -e "${RED}AlphaTrader exited with error code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE