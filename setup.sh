#!/bin/bash

# Options Trading System - Environment Setup Script
# Phase 1: Core Infrastructure (Day 1-2)

echo "================================================"
echo "Options Trading System - Environment Setup"
echo "Phase 1: Core Infrastructure"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1 successful"
    else
        echo -e "${RED}✗${NC} $1 failed"
        exit 1
    fi
}

# 1. Check Python version
echo -e "\n${YELLOW}Step 1: Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | sed 's/Python //' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓${NC} Python $python_version meets minimum requirement (3.11+)"
else
    echo -e "${RED}✗${NC} Python $python_version does not meet minimum requirement (3.11+)"
    exit 1
fi

# 2. Create Python virtual environment
echo -e "\n${YELLOW}Step 2: Creating Python virtual environment...${NC}"
python3 -m venv venv
check_status "Virtual environment creation"

# 3. Activate virtual environment
echo -e "\n${YELLOW}Step 3: Activating virtual environment...${NC}"
source venv/bin/activate
check_status "Virtual environment activation"

# 4. Upgrade pip
echo -e "\n${YELLOW}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip
check_status "Pip upgrade"

# 5. Install Python packages
echo -e "\n${YELLOW}Step 5: Installing Python packages...${NC}"
pip install -r requirements.txt
check_status "Python packages installation"

# 6. Check Redis installation
echo -e "\n${YELLOW}Step 6: Checking Redis installation...${NC}"
if command -v redis-server &> /dev/null; then
    redis_version=$(redis-server --version | sed 's/.*v=//' | cut -d' ' -f1)
    echo -e "${GREEN}✓${NC} Redis $redis_version is installed"
else
    echo -e "${RED}✗${NC} Redis is not installed"
    echo "Please install Redis 7.0+ using:"
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
    echo "  Or visit: https://redis.io/download"
fi

# 7. Create necessary directories
echo -e "\n${YELLOW}Step 7: Creating project directories...${NC}"
mkdir -p logs
mkdir -p data
mkdir -p temp
check_status "Directory creation"

# 8. Create .env file if it doesn't exist
echo -e "\n${YELLOW}Step 8: Setting up environment variables...${NC}"
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    echo "# Alpha Vantage Configuration
AV_API_KEY=your_alpha_vantage_api_key_here
AV_BASE_URL=https://www.alphavantage.co/query
AV_RATE_LIMIT=600

# Interactive Brokers Configuration
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_ACCOUNT=DU1234567

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_MAX_MEMORY=4gb

# Trading Configuration
TRADING_MODE=paper
MAX_POSITIONS=5
MAX_POSITION_SIZE_PCT=0.25
MAX_DAILY_LOSS=2000
RISK_PER_TRADE=0.02

# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
WS_PORT=8001

# Dashboard Configuration
DASHBOARD_PORT=8080
DASHBOARD_HOST=localhost

# Environment
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log" > .env
    echo -e "${GREEN}✓${NC} .env file created"
    echo -e "${YELLOW}⚠${NC} Please update .env with your API keys and account details"
else
    echo -e "${GREEN}✓${NC} .env file already exists"
fi

# 9. Display next steps
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "\nNext steps:"
echo -e "1. ${YELLOW}Update .env file${NC} with your actual API keys:"
echo -e "   - Alpha Vantage API key"
echo -e "   - IBKR account number"
echo -e ""
echo -e "2. ${YELLOW}Start Redis server${NC}:"
echo -e "   redis-server --maxmemory 4gb --maxmemory-policy volatile-lru"
echo -e ""
echo -e "3. ${YELLOW}Start IB Gateway/TWS${NC}:"
echo -e "   - Open IB Gateway or Trader Workstation"
echo -e "   - Enable API connections"
echo -e "   - Set socket port to 7497 (paper) or 7496 (live)"
echo -e ""
echo -e "4. ${YELLOW}Test connections${NC}:"
echo -e "   python scripts/test_connections.py"
echo -e ""
echo -e "5. ${YELLOW}Run health check${NC}:"
echo -e "   python scripts/health_check.py"