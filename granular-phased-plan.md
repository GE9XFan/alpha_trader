# Granular Phased Implementation Plan - Version 2.0
**Approach:** Skeleton-First, API-Driven, Configuration-Based  
**Key Change:** Build complete skeleton first, then implement APIs one-by-one with schema evolution

---

## **Phase 0: Infrastructure Foundation & Complete Skeleton**
Build the entire project skeleton with empty modules and configuration structure.

### **Step 0.1: Project Structure & Version Control**
**Objective**: Create complete project skeleton with all modules as empty shells.
**Key Files to Create**:
```
/
├── src/
│   ├── foundation/
│   │   ├── __init__.py
│   │   ├── config_manager.py      # ConfigManager class skeleton
│   │   ├── base_module.py         # BaseModule abstract class
│   │   └── exceptions.py          # Custom exceptions
│   ├── connections/
│   │   ├── __init__.py
│   │   ├── base_client.py         # BaseAPIClient abstract class
│   │   ├── ibkr_connection.py     # IBKRConnectionManager skeleton
│   │   └── av_client.py           # AlphaVantageClient skeleton
│   ├── data/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py        # TokenBucketRateLimiter skeleton
│   │   ├── scheduler.py           # DataScheduler skeleton
│   │   ├── ingestion.py           # DataIngestionPipeline skeleton
│   │   ├── cache_manager.py       # CacheManager skeleton
│   │   └── schema_builder.py      # SchemaBuilder skeleton
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── indicator_processor.py # IndicatorProcessor skeleton
│   │   ├── greeks_validator.py    # GreeksValidator skeleton
│   │   └── analytics_engine.py    # AnalyticsEngine skeleton
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── feature_builder.py     # FeatureBuilder skeleton
│   │   ├── model_suite.py         # ModelSuite skeleton
│   │   └── utils.py               # ML utilities skeleton
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── decision_engine.py     # DecisionEngine skeleton
│   │   └── strategy_engine.py     # StrategyEngine skeleton
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py       # BaseStrategy abstract class
│   │   ├── zero_dte.py           # ZeroDTEStrategy skeleton
│   │   ├── one_dte.py            # OneDTEStrategy skeleton
│   │   ├── swing_14d.py          # Swing14DStrategy skeleton
│   │   └── moc_imbalance.py      # MOCImbalanceStrategy skeleton
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── risk_manager.py        # RiskManager skeleton
│   │   └── position_sizer.py      # PositionSizer skeleton
│   ├── execution/
│   │   ├── __init__.py
│   │   └── ibkr_executor.py       # IBKRExecutor skeleton
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── trade_monitor.py       # TradeMonitor skeleton
│   ├── publishing/
│   │   ├── __init__.py
│   │   └── publisher.py           # DiscordPublisher skeleton
│   └── api/
│       ├── __init__.py
│       └── dashboard_api.py       # DashboardAPI skeleton
├── config/                         # Configuration structure
│   ├── .env.example
│   ├── system/
│   ├── apis/
│   ├── data/
│   ├── strategies/
│   ├── risk/
│   ├── ml/
│   ├── execution/
│   ├── monitoring/
│   └── environments/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── system/
├── scripts/
│   ├── health_check.py
│   ├── test_api.py
│   └── backup_db.py
├── models/                         # ML model storage
├── logs/                          # Log files
├── reports/                       # Generated reports
├── .gitignore
├── requirements.txt
└── README.md
```

**Actionable Tasks**:
1. Create all directories
2. Create all Python files with class skeletons (empty methods with `pass`)
3. Create all `__init__.py` files with proper imports
4. Set up `.gitignore` for Python project
5. Initialize Git repository
6. Make initial commit

**Definition of Done**: 
- All files exist with proper class/function signatures
- No import errors when importing any module
- Project structure matches specification exactly

### **Step 0.2: Configuration Structure**
**Objective**: Create complete configuration file structure with templates.
**Key Files to Create**:
- `config/.env.example` with all required keys
- All YAML configuration templates in respective directories
- Example configuration for each component

**Actionable Tasks**:
1. Create `.env.example` with placeholders for:
   - `AV_API_KEY`
   - `IBKR_USERNAME`
   - `IBKR_PASSWORD`
   - `IBKR_ACCOUNT`
   - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
   - `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
   - `DISCORD_WEBHOOK_URL`
2. Create YAML templates for each configuration category
3. Set sensible defaults for all parameters

**Definition of Done**: 
- All configuration files have templates
- Configuration structure documented
- Can load configuration without errors

### **Step 0.3: Virtual Environment & Dependencies**
**Objective**: Set up Python environment with all required packages.
**Actionable Tasks**:
1. Create virtual environment: `python3.11 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install core dependencies:
   ```bash
   pip install psycopg2-binary redis sqlalchemy pydantic
   pip install aiohttp requests pandas numpy
   pip install ib_insync python-dotenv pyyaml
   pip install fastapi uvicorn websockets discord-webhook
   pip install pytest pytest-asyncio pytest-cov
   pip install scikit-learn joblib
   ```
4. Generate `requirements.txt`: `pip freeze > requirements.txt`

**Definition of Done**: 
- Virtual environment activated
- All packages installed
- `requirements.txt` committed

### **Step 0.4: Database Setup (System Tables Only)**
**Objective**: Initialize PostgreSQL with system tables only (no data tables yet).
**Key Files to Create**:
- `scripts/init_system_db.sql`

**SQL to Execute**:
```sql
-- System configuration
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API call logging
CREATE TABLE IF NOT EXISTS api_call_log (
    id SERIAL PRIMARY KEY,
    api_name VARCHAR(50),
    endpoint VARCHAR(100),
    parameters JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    error_message TEXT,
    called_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema versions for migrations
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    api_name VARCHAR(50),
    migration_sql TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Emergency log
CREATE TABLE IF NOT EXISTS emergency_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50),
    description TEXT,
    action_taken TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Actionable Tasks**:
1. Install PostgreSQL if not installed
2. Create database: `createdb trading_system`
3. Execute system tables SQL
4. Test connection from Python

**Definition of Done**: 
- PostgreSQL running
- System tables created
- Can connect and query from Python

### **Step 0.5: Redis Setup**
**Objective**: Initialize Redis for caching.
**Actionable Tasks**:
1. Install Redis if not installed
2. Start Redis server
3. Test connection: `redis-cli ping`
4. Test from Python using redis-py

**Definition of Done**: 
- Redis running
- Can connect from Python
- Basic set/get operations work

### **Step 0.6: Implement ConfigManager**
**Objective**: Build the configuration management system.
**Key Implementation**:
```python
# src/foundation/config_manager.py
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.config = {}
        self.config_dir = Path('config')
        self._load_env_variables()
        self._load_yaml_configs()
        self._apply_environment_overrides()
    
    def _load_env_variables(self):
        load_dotenv()
        self.config['env'] = {
            'av_api_key': os.getenv('AV_API_KEY'),
            'ibkr_username': os.getenv('IBKR_USERNAME'),
            # ... all env variables
        }
    
    def _load_yaml_configs(self):
        # Load all YAML files from config directories
        pass
    
    def get(self, path: str, default: Any = None) -> Any:
        # Get config value by dot notation
        pass
```

**Actionable Tasks**:
1. Implement full ConfigManager class
2. Test loading all configuration files
3. Test dot notation access
4. Add validation for required keys

**Definition of Done**: 
- ConfigManager loads all configs
- Can access any config value
- Validates required keys exist

### **Step 0.7: Implement Base Classes**
**Objective**: Create abstract base classes for consistent module interface.
**Key Classes**:
- `BaseModule`: Abstract base for all modules
- `BaseAPIClient`: Abstract base for API clients  
- `BaseStrategy`: Abstract base for strategies

**Actionable Tasks**:
1. Implement BaseModule with initialize(), health_check(), shutdown()
2. Implement BaseAPIClient with connect(), disconnect(), call()
3. Implement BaseStrategy with evaluate(), execute()
4. Add dependency injection support

**Definition of Done**: 
- All base classes implemented
- Can be inherited by concrete classes
- Dependency management works

### **Step 0.8: Create Module Initialization Chain**
**Objective**: Build the system that initializes all modules in correct order.
**Key Implementation**:
```python
# scripts/initialize_system.py
from src.foundation.config_manager import ConfigManager

INITIALIZATION_ORDER = [
    'ConfigManager',
    'Database',
    'Redis', 
    'RateLimiter',
    'AlphaVantageClient',
    'IBKRConnection',
    # ... full order
]

def initialize_system():
    modules = {}
    config = ConfigManager()
    
    for module_name in INITIALIZATION_ORDER:
        # Initialize each module
        pass
    
    return modules
```

**Actionable Tasks**:
1. Create initialization script
2. Implement module loading logic
3. Add health checks after each module
4. Test full initialization chain

**Definition of Done**: 
- All modules initialize in order
- Health checks pass
- No circular dependencies

---

## **Phase 0.5: API Discovery & Schema Evolution**
Test each API individually and build schema based on actual responses.

### **Step 0.5.1: Create API Testing Framework**
**Objective**: Build utilities to test and document each API.
**Key Files**:
- `scripts/test_api.py`
- `src/data/schema_builder.py`

**Implementation**:
```python
# scripts/test_api.py
class APITester:
    def test_endpoint(self, api_name: str, params: dict):
        # 1. Make API call
        # 2. Log full response
        # 3. Analyze structure
        # 4. Generate schema
        # 5. Create table
        # 6. Test ingestion
        pass
```

**Actionable Tasks**:
1. Implement APITester class
2. Add response logging
3. Add schema generation
4. Add table creation
5. Test with one API

**Definition of Done**: 
- Can test any API endpoint
- Generates correct schema
- Creates appropriate table

### **Step 0.5.2: Test Alpha Vantage APIs (One by One)**
**Objective**: Test all 43 Alpha Vantage APIs in priority order.

**Priority Order**:
1. **REALTIME_OPTIONS** (Most Critical)
2. **RSI**
3. **MACD**
4. **BBANDS**
5. **ATR**
6. **VWAP**
7. ... (continue through all 43)

**For EACH API**:
```python
# Workflow for each API
API_WORKFLOW = {
    "1": "Update av_client.py with endpoint method",
    "2": "Make test call with SPY",
    "3": "Log complete response structure",
    "4": "Analyze data types and structure",
    "5": "Design optimal table schema",
    "6": "Create table in database",
    "7": "Implement ingestion logic",
    "8": "Test data persistence",
    "9": "Verify data retrieval",
    "10": "Document any quirks",
    "11": "Add rate limit handling",
    "12": "Create unit tests",
    "13": "Update configuration",
    "14": "Commit and document"
}
```

**Example for REALTIME_OPTIONS**:
1. Add method to `av_client.py`:
   ```python
   def get_realtime_options(self, symbol: str):
       params = {
           'function': 'REALTIME_OPTIONS',
           'symbol': symbol,
           'apikey': self.api_key
       }
       return self._make_request(params)
   ```

2. Test and analyze response
3. Create table based on actual response:
   ```sql
   CREATE TABLE av_realtime_options (
       -- columns based on actual response
   );
   ```

4. Implement ingestion in `ingestion.py`
5. Test full flow
6. Document findings

**Definition of Done (per API)**: 
- API method implemented and tested
- Table created with correct schema
- Data successfully persisted
- Can query data back
- Quirks documented

### **Step 0.5.3: Test IBKR Data Feeds**
**Objective**: Test all IBKR data feeds and create schemas.

**Data Feeds to Test**:
1. Real-time bars (1s, 5s, 1m, 5m)
2. Real-time quotes
3. MOC imbalance data
4. Account data

**Actionable Tasks**:
1. Connect to paper trading account
2. Subscribe to each data type
3. Log data structure
4. Create appropriate tables
5. Test ingestion
6. Verify data flow

**Definition of Done**: 
- All IBKR feeds tested
- Tables created for each feed
- Data flowing correctly

---

## **Phase 1: Complete Connections Layer**
With skeleton in place and APIs tested, implement full connection logic.

### **Step 1.1: Implement TokenBucketRateLimiter**
**Objective**: Build robust rate limiting for Alpha Vantage.
**Implementation Focus**:
- 600 calls/minute hard limit
- 10 tokens/second refill
- Burst capability to 20
- Thread-safe implementation

**Definition of Done**: 
- Rate limiter prevents exceeding limits
- Handles bursts correctly
- Thread-safe operation verified

### **Step 1.2: Complete AlphaVantageClient**
**Objective**: Finalize all 43 API endpoint implementations.
**Tasks**:
- Add retry logic with exponential backoff
- Add comprehensive error handling
- Integrate rate limiter
- Add response caching where appropriate

**Definition of Done**: 
- All 43 endpoints fully functional
- Rate limiting integrated
- Error handling comprehensive
- All unit tests pass

### **Step 1.3: Complete IBKRConnectionManager**
**Objective**: Finalize IBKR connection with all features.
**Tasks**:
- Implement reconnection logic
- Add subscription management
- Handle all data types
- Implement order execution methods

**Definition of Done**: 
- Stable connection maintained
- All data feeds working
- Can execute orders (paper)
- Handles disconnections gracefully

---

## **Phase 2: Data Management Layer**
Implement scheduling, ingestion, and caching.

### **Step 2.1: Implement DataScheduler**
**Objective**: Build the orchestrator for all API calls.
**Key Features**:
- Tier-based scheduling (A, B, C)
- Dynamic priority adjustment
- MOC window handling
- Resource optimization

**Definition of Done**: 
- Schedules respect tier priorities
- Stays under rate limits
- MOC window elevates correctly

### **Step 2.2: Complete DataIngestionPipeline**
**Objective**: Build robust data normalization and storage.
**Key Features**:
- Normalize all API responses
- Handle different data formats
- Validate before storage
- Log all operations

**Definition of Done**: 
- All data types handled
- Validation working
- Error handling complete
- Performance optimized

### **Step 2.3: Implement CacheManager**
**Objective**: Build Redis-based caching layer.
**Key Features**:
- TTL-based expiration
- Cache warming
- Cache invalidation
- Memory management

**Definition of Done**: 
- Caching reduces API calls
- TTL correctly enforced
- Memory usage stable

---

## **Phase 3: Analytics Engine**
Process indicators and validate Greeks.

### **Step 3.1: Implement IndicatorProcessor**
**Objective**: Build indicator aggregation and processing.
**Definition of Done**: 
- Retrieves all indicators
- Calculates derived metrics
- Handles missing data

### **Step 3.2: Implement GreeksValidator**
**Objective**: Validate all Greeks data.
**Critical Checks**:
- Delta: -1 to 1
- Gamma: >= 0
- Theta: sign appropriate
- Data freshness < 30s

**Definition of Done**: 
- All validation rules implemented
- Rejects invalid data
- Logs all rejections

### **Step 3.3: Build AnalyticsEngine**
**Objective**: Calculate derived analytics.
**Definition of Done**: 
- All calculations correct
- Performance optimized
- Results cached appropriately

---

## **Phase 4: ML Layer**
Integrate frozen models and feature engineering.

### **Step 4.1: Implement FeatureBuilder**
**Objective**: Build feature engineering pipeline.
**Definition of Done**: 
- Extracts all features
- Handles missing values
- Scales appropriately
- Deterministic output

### **Step 4.2: Implement ModelSuite**
**Objective**: Load and run frozen models.
**Note**: Models must be pre-trained and provided.
**Definition of Done**: 
- Models load correctly
- Predictions generated
- Confidence scores calculated

---

## **Phase 5: Decision Engine**
Implement strategies and decision logic.

### **Step 5.1: Implement All Strategies**
**Strategies to Implement**:
1. ZeroDTEStrategy
2. OneDTEStrategy  
3. Swing14DStrategy
4. MOCImbalanceStrategy

**For Each Strategy**:
- Load parameters from config
- Implement evaluation logic
- Calculate confidence scores
- Generate trade signals

**Definition of Done**: 
- All strategies implemented
- Use configuration values
- Generate correct signals
- Logging comprehensive

### **Step 5.2: Implement DecisionEngine**
**Objective**: Build master decision maker.
**Definition of Done**: 
- Integrates all inputs
- Selects appropriate strategy
- Makes final decision
- Logs all decisions

---

## **Phase 6: Risk & Execution**
**CRITICAL**: Paper trading only until Phase 9.

### **Step 6.1: Implement RiskManager**
**Objective**: Build comprehensive risk checks.
**Key Checks**:
- Position Greeks limits
- Portfolio Greeks limits
- Capital limits
- Correlation limits

**Definition of Done**: 
- All limits enforced
- Violations logged
- Can override manually

### **Step 6.2: Implement PositionSizer**
**Objective**: Calculate appropriate position sizes.
**Definition of Done**: 
- Sizes based on risk
- Respects account limits
- Kelly criterion implemented

### **Step 6.3: Implement IBKRExecutor**
**Objective**: Execute trades via IBKR.
**CRITICAL**: Paper account only!
**Definition of Done**: 
- Orders submitted correctly
- Fills confirmed
- Positions tracked

---

## **Phase 7: Output Layer**
Build monitoring and alerting.

### **Step 7.1: Implement DiscordPublisher**
**Objective**: Send alerts to Discord.
**Definition of Done**: 
- Formatted messages sent
- All event types handled
- Rate limiting respected

### **Step 7.2: Build Dashboard API**
**Objective**: Create monitoring dashboard.
**Endpoints**:
- `/health`
- `/positions`
- `/performance`
- `/stream` (WebSocket)

**Definition of Done**: 
- All endpoints working
- Real-time updates via WebSocket
- Performance acceptable

---

## **Phase 8: Integration Testing**
Test complete system end-to-end.

### **Step 8.1: 5-Day Paper Trading Test**
**Objective**: Run system for 5 full trading days.
**Success Criteria**:
- No unhandled exceptions
- Win rate > 45%
- All strategies execute
- Risk limits respected

### **Step 8.2: Chaos Testing**
**Tests to Perform**:
- Kill database mid-trade
- Block API endpoints
- Disconnect IBKR
- Exhaust rate limits
- Fill disk space

**Definition of Done**: 
- System recovers from all failures
- No data corruption
- Positions always reconciled

---

## **Phase 9: Production Deployment**
Go live with real money (carefully).

### **Step 9.1: Complete Go/No-Go Checklist**
**All Must Be True**:
- [ ] 5-day paper test successful
- [ ] Win rate > 45%
- [ ] Profit factor > 1.2
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Backups tested
- [ ] Emergency procedures tested
- [ ] Configuration externalized
- [ ] Monitoring operational

### **Step 9.2: Initial Deployment**
**Week 1**: $1,000 capital limit
**Week 2**: $2,500 if profitable
**Week 3**: $5,000 if profitable
**Week 4**: $10,000 target

### **Step 9.3: Production Monitoring**
- Review every trade for first week
- Daily performance reviews
- Gradual parameter tuning
- Continuous improvement

**Definition of Done**: 
- System trading profitably
- Stable for 30 days
- All metrics within targets

---

## **Critical Success Factors**

1. **Never skip API testing** - Each API must be fully tested before moving on
2. **Schema matches reality** - Tables must match actual API responses exactly
3. **Configuration-driven** - No hard-coded values anywhere
4. **Paper test thoroughly** - Minimum 5 days before real money
5. **Document everything** - Every quirk, every issue, every decision

---

## **Timeline Estimate**

- **Week 1**: Phase 0 (Infrastructure & Skeleton)
- **Week 2-3**: Phase 0.5 (API Discovery)
- **Week 4**: Phase 1 (Connections)
- **Week 5**: Phase 2 (Data Management)
- **Week 6**: Phase 3 (Analytics)
- **Week 7**: Phase 4-5 (ML & Decision)
- **Week 8**: Phase 6 (Risk & Execution)
- **Week 9**: Phase 7 (Output)
- **Week 10-11**: Phase 8 (Integration Testing)
- **Week 12+**: Phase 9 (Production)