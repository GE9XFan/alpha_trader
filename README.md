# Trading System

Production-grade automated options trading system.

## Architecture

- **Skeleton-First Development**: Complete module structure before implementation
- **API-Driven Schema**: Database schema evolves based on actual API responses
- **Configuration-Driven**: All parameters externalized to YAML files
- **No Hardcoded Values**: Everything configurable

## Project Structure

```
src/
├── foundation/     # Core infrastructure
├── connections/    # API connections (IBKR, Alpha Vantage)
├── data/          # Data management layer
├── analytics/     # Analytics and indicators
├── ml/            # Machine learning components
├── decision/      # Decision engine
├── strategies/    # Trading strategies
├── risk/          # Risk management
├── execution/     # Order execution
├── monitoring/    # Trade monitoring
├── publishing/    # Alert publishing
└── api/           # Dashboard API
```

## Development Phases

1. **Phase 0**: Infrastructure & Skeleton
2. **Phase 0.5**: API Discovery & Schema Evolution
3. **Phase 1**: Complete Connections Layer
4. **Phase 2**: Data Management Layer
5. **Phase 3**: Analytics Engine
6. **Phase 4**: ML Layer
7. **Phase 5**: Decision Engine
8. **Phase 6**: Risk & Execution
9. **Phase 7**: Output Layer
10. **Phase 8**: Integration Testing
11. **Phase 9**: Production Deployment

## Quick Start

1. Clone repository
2. Create virtual environment: `python3.11 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure
6. Initialize database: `python scripts/init_system_db.py`
7. Run health check: `python scripts/health_check.py`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Documentation

See `docs/` directory for detailed documentation.

## License

Proprietary - All Rights Reserved
