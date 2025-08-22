# API Implementation Framework - Universal Steps
**Purpose:** Standardized approach for integrating any external API  
**Duration:** ~2-3 hours per API endpoint  
**Outcome:** Production-ready API integration with caching, scheduling, and monitoring

## Step 1: API Discovery & Documentation (30-45 mins)

### 1.1 Create Test Script
**Location:** `scripts/test_[api_name]_api.py`
```python
#!/usr/bin/env python3
"""Test [API_NAME] API and document response structure"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.foundation.config_manager import ConfigManager
import requests

def test_api():
    config = ConfigManager()
    params = {
        # API-specific parameters
        'endpoint': '[ENDPOINT]',
        'apikey': config.[api]_key,
        'format': 'json'
    }
    # Make API call and save response
    # Document structure, rate limits, data types
```

### 1.2 API Response Analysis
**Actions:**
- Execute test script
- Save raw response to `data/api_responses/[api]_[timestamp].json`
- Document structure:
  - Top-level keys and their types
  - Data point count
  - Date/time formats
  - Value formats (strings, numbers, nested objects)
  - Error response formats
  - Rate limit headers

### 1.3 Create API Documentation
**File:** `docs/apis/[api_name]_spec.md`
- Endpoint URLs
- Required/optional parameters
- Response structure
- Rate limits
- Error codes
- Data types and ranges

## Step 2: Configuration Setup (15-20 mins)

### 2.1 API Configuration
**File:** `config/apis/[api_name].yaml`
```yaml
base_url: "[API_BASE_URL]"
auth_method: "[key|oauth|basic]"
rate_limits:
  calls_per_minute: [NUMBER]
  calls_per_day: [NUMBER]

endpoints:
  [endpoint_name]:
    path: "[/path/to/endpoint]"
    method: "[GET|POST]"
    cache_ttl: [SECONDS]
    retry_strategy:
      max_retries: 3
      backoff_factor: 2
    default_params:
      param1: "value1"
      param2: "value2"
```

### 2.2 Scheduler Configuration
**File:** `config/data/schedules.yaml`
```yaml
[schedule_group]:
  apis: ["[API_NAME]"]
  priority_1_interval: [SECONDS]
  priority_2_interval: [SECONDS]
  priority_3_interval: [SECONDS]
  calls_per_resource: [NUMBER]
  enabled: true
```

### 2.3 Environment Variables
**File:** `.env`
```bash
[API_NAME]_API_KEY=your_key_here
[API_NAME]_BASE_URL=https://api.example.com
[API_NAME]_TIMEOUT=30
```

## Step 3: Client Method Implementation (30-45 mins)

### 3.1 Remove Hardcoded Values
**Critical Checklist:**
- [ ] No hardcoded URLs
- [ ] No hardcoded API keys
- [ ] No hardcoded default parameters
- [ ] No hardcoded timeout values
- [ ] All values from configuration

### 3.2 Implement Client Method
**File:** `src/connections/[api]_client.py`
```python
class [API]Client:
    def __init__(self):
        self.config = ConfigManager()
        self.cache = CacheManager()
        self.setup_rate_limiter()
        
    def get_[endpoint](self, resource_id, **kwargs):
        """
        Get [data_type] from [API_NAME]
        
        Args:
            resource_id: Identifier for the resource
            **kwargs: Override default parameters
            
        Returns:
            dict: API response or cached data
        """
        # Load endpoint configuration
        endpoint_config = self.config.[api]_config['endpoints']['[endpoint]']
        
        # Build parameters from config, no hardcoding
        params = self._build_params(endpoint_config, resource_id, **kwargs)
        
        # Check cache first
        cache_key = self._generate_cache_key(params)
        if use_cache and (cached := self.cache.get(cache_key)):
            return cached
            
        # Make API call with retry logic
        response = self._make_request(params)
        
        # Cache successful response
        if response:
            self.cache.set(cache_key, response, ttl=endpoint_config['cache_ttl'])
            
        return response
```

### 3.3 Add Error Handling
```python
def _make_request(self, params):
    try:
        # Rate limiting check
        if not self.rate_limiter.allow_request():
            raise RateLimitExceeded()
            
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        # Log successful call
        self._log_api_call(success=True)
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        self._log_api_call(success=False, error=str(e))
        return None
```

### 3.4 Client Testing
**Script:** `scripts/test_[api]_client.py`
- Test normal operation
- Test caching behavior
- Test error handling
- Test rate limiting
- Measure performance (API vs cache)

## Step 4: Schema Design & Database Setup (30 mins)

### 4.1 Design Schema Based on API Response
**File:** `scripts/create_[api]_table.sql`
```sql
CREATE TABLE IF NOT EXISTS [api]_[endpoint] (
    id SERIAL PRIMARY KEY,
    resource_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Data fields based on API response
    field1 [DATATYPE],
    field2 [DATATYPE],
    json_data JSONB,  -- For flexible/nested data
    
    -- Metadata
    api_version VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(resource_id, timestamp),
    CHECK (field1 >= 0)  -- Add data validation
);

-- Performance indexes
CREATE INDEX idx_[table]_resource_timestamp 
    ON [table](resource_id, timestamp DESC);
CREATE INDEX idx_[table]_timestamp 
    ON [table](timestamp DESC);
CREATE INDEX idx_[table]_json_data 
    ON [table] USING GIN(json_data);  -- For JSONB queries
```

### 4.2 Create Migration
**File:** `migrations/[timestamp]_create_[api]_tables.sql`
- Include rollback statements
- Add comments explaining fields
- Include sample data for testing

### 4.3 Execute Migration
```bash
# Run migration
psql -U [user] -d [database] -f migrations/[timestamp]_create_[api]_tables.sql

# Verify structure
psql -U [user] -d [database] -c "\d [table_name]"
```

## Step 5: Ingestion Pipeline Implementation (45-60 mins)

### 5.1 Implement Ingestion Method
**File:** `src/data/ingestion.py`
```python
def ingest_[api]_data(self, api_response, resource_id, **metadata):
    """
    Ingest [API] data into database
    
    Args:
        api_response: Raw API response
        resource_id: Resource identifier
        **metadata: Additional context (interval, version, etc.)
    
    Returns:
        int: Number of records processed
    """
    # Validate response
    if not self._validate_response(api_response):
        logger.error(f"Invalid response for {resource_id}")
        return 0
    
    # Extract data based on API structure
    data_points = self._extract_data_points(api_response)
    logger.info(f"Processing {len(data_points)} records for {resource_id}")
    
    records_processed = 0
    batch = []
    
    for point in data_points:
        # Transform data
        record = self._transform_record(point, resource_id, **metadata)
        
        # Add to batch
        batch.append(record)
        
        # Process batch when full
        if len(batch) >= self.batch_size:
            records_processed += self._process_batch(batch)
            batch = []
    
    # Process remaining records
    if batch:
        records_processed += self._process_batch(batch)
    
    # Update cache after successful ingestion
    self._update_ingestion_cache(resource_id, api_response)
    
    return records_processed
```

### 5.2 Implement Data Transformation
```python
def _transform_record(self, raw_data, resource_id, **metadata):
    """Transform API data to database format"""
    return {
        'resource_id': resource_id,
        'timestamp': self._parse_timestamp(raw_data.get('time')),
        'field1': self._to_decimal(raw_data.get('value1')),
        'field2': self._to_integer(raw_data.get('value2')),
        'json_data': json.dumps(raw_data.get('extra', {})),
        'api_version': metadata.get('version', '1.0'),
        'updated_at': datetime.now()
    }
```

### 5.3 Implement Batch Processing
```python
def _process_batch(self, batch):
    """Process batch with upsert logic"""
    try:
        # Use COPY for initial load, INSERT ON CONFLICT for updates
        stmt = """
            INSERT INTO [table] (resource_id, timestamp, field1, field2, json_data)
            VALUES (%(resource_id)s, %(timestamp)s, %(field1)s, %(field2)s, %(json_data)s)
            ON CONFLICT (resource_id, timestamp) 
            DO UPDATE SET
                field1 = EXCLUDED.field1,
                field2 = EXCLUDED.field2,
                updated_at = CURRENT_TIMESTAMP
        """
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(stmt, batch)
            conn.commit()
            return len(batch)
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 0
```

### 5.4 Pipeline Testing
**Script:** `scripts/test_[api]_pipeline.py`
- Test end-to-end flow
- Verify data transformation
- Check duplicate handling
- Validate database constraints
- Measure throughput

## Step 6: Scheduler Integration (30-45 mins)

### 6.1 Add Fetch Method
**File:** `src/data/scheduler.py`
```python
def _fetch_[api]_[endpoint](self, resource_id, **params):
    """
    Fetch [endpoint] data for resource
    Called by scheduler based on configuration
    """
    # Check operational constraints
    if not self._should_fetch(resource_id):
        logger.info(f"Skipping {resource_id} - constraints not met")
        return
    
    # Initialize clients
    api_client = [API]Client()
    ingestion = DataIngestion()
    
    try:
        # Fetch data with configured parameters
        data = api_client.get_[endpoint](resource_id, **params)
        
        if data:
            # Ingest into database
            records = ingestion.ingest_[api]_data(data, resource_id, **params)
            logger.info(f"✓ {resource_id}: {records} records processed")
            
            # Update metrics
            self._update_metrics('success', resource_id, records)
        else:
            self._update_metrics('no_data', resource_id, 0)
            
    except Exception as e:
        logger.error(f"Failed to fetch {resource_id}: {e}")
        self._update_metrics('error', resource_id, 0)
```

### 6.2 Add Scheduling Method
```python
def _schedule_[api]_jobs(self):
    """Schedule [API] data collection jobs"""
    
    # Load configuration
    schedule_config = self.config.schedules.get('[schedule_group]', {})
    
    if '[API]' not in schedule_config.get('apis', []):
        logger.info("[API] not configured in schedule group")
        return
    
    # Get resources to track
    resources = self._get_resources_by_priority()
    
    # Schedule by priority tier
    for priority, resource_list in resources.items():
        interval = schedule_config[f'priority_{priority}_interval']
        
        for resource_id in resource_list:
            job_id = f"[api]_{resource_id}_{priority}"
            
            self.scheduler.add_job(
                func=self._fetch_[api]_[endpoint],
                trigger='interval',
                seconds=interval,
                id=job_id,
                args=[resource_id],
                kwargs=self._get_job_params(resource_id),
                max_instances=1,
                replace_existing=True
            )
            
    logger.info(f"Scheduled {len(self.scheduler.get_jobs())} [API] jobs")
```

### 6.3 Update Main Job Creation
```python
def _create_jobs(self):
    """Create all scheduled jobs"""
    
    # Existing jobs...
    
    # Add new API jobs
    if '[schedule_group]' in self.api_groups:
        self._schedule_[api]_jobs()
    
    # Log summary
    self._log_job_summary()
```

### 6.4 Scheduler Testing
**Script:** `scripts/test_[api]_scheduler.py`
- Verify job creation
- Test execution intervals
- Check resource prioritization
- Monitor API rate limits
- Validate error recovery

## Step 7: End-to-End Testing (30-45 mins)

### 7.1 Comprehensive System Test
**Script:** `scripts/test_[api]_complete.py`
```python
def test_complete_system():
    """Full system integration test"""
    
    # 1. Clear test data
    cleanup_test_data()
    
    # 2. Initialize components
    api_client = [API]Client()
    scheduler = DataScheduler()
    ingestion = DataIngestion()
    
    # 3. Test API connectivity
    assert test_api_connection(api_client)
    
    # 4. Test single resource flow
    test_resource = 'TEST_001'
    data = api_client.get_[endpoint](test_resource)
    records = ingestion.ingest_[api]_data(data, test_resource)
    assert records > 0
    
    # 5. Test scheduler with multiple resources
    scheduler.start()
    time.sleep(120)  # Run for 2 minutes
    scheduler.stop()
    
    # 6. Validate database state
    stats = get_database_statistics()
    assert stats['total_records'] > 0
    assert stats['unique_resources'] >= len(test_resources)
    
    # 7. Test data quality
    validate_data_quality()
    
    # 8. Performance metrics
    print_performance_summary()
```

### 7.2 Data Quality Validation
```python
def validate_data_quality():
    """Validate ingested data quality"""
    
    checks = {
        'nulls': check_for_nulls(),
        'duplicates': check_for_duplicates(),
        'ranges': validate_value_ranges(),
        'timestamps': validate_timestamp_continuity(),
        'relationships': validate_data_relationships()
    }
    
    for check_name, result in checks.items():
        assert result['passed'], f"Failed {check_name}: {result['message']}"
```

### 7.3 Performance Benchmarks
- API calls per minute: [TARGET]
- Cache hit rate: > 80%
- Ingestion throughput: > 1000 records/second
- Database query time: < 100ms
- Memory usage: < 500MB

## Step 8: Documentation & Deployment

### 8.1 Create Documentation
**Files to create:**
1. `docs/apis/[api]_integration.md` - Integration guide
2. `docs/apis/[api]_troubleshooting.md` - Common issues
3. `scripts/test_[api]_*.py` - Test suite
4. `scripts/create_[api]_tables.sql` - Database schema
5. `monitoring/[api]_dashboard.json` - Monitoring config

### 8.2 Update System Documentation
**Files to update:**
1. `README.md` - Add API to supported integrations
2. `docs/configuration.md` - Document new config options
3. `docs/api_limits.md` - Update rate limit tracking
4. `CHANGELOG.md` - Document changes

### 8.3 Git Commit Template
```bash
git add -A
git commit -m "[API_NAME] Integration: [Brief Summary]

- Implemented [endpoint] with full scheduler integration
- Configuration-driven parameters (no hardcoding)
- Database schema supports [key features]
- Scheduler manages [X] jobs across [Y] priority tiers
- Caching with [TTL]s TTL for [use case]
- API usage: [X%] of rate limit capacity
- Test coverage: [X%]
- Performance: [key metrics]

Resolves: #[issue_number]"
```

## Quality Checklist

### Code Quality
- [ ] No hardcoded values
- [ ] All parameters from configuration
- [ ] Comprehensive error handling
- [ ] Logging at appropriate levels
- [ ] Type hints on all methods
- [ ] Docstrings with examples

### Testing
- [ ] Unit tests for each component
- [ ] Integration tests for pipeline
- [ ] End-to-end system test
- [ ] Performance benchmarks met
- [ ] Error scenarios tested
- [ ] Rate limit compliance verified

### Operations
- [ ] Monitoring configured
- [ ] Alerts defined
- [ ] Runbook created
- [ ] Rollback procedure documented
- [ ] Resource limits set
- [ ] Backup strategy defined

## Common Pitfalls to Avoid

1. **Hardcoded Values:** Always use configuration
2. **Missing Error Handling:** Expect API failures
3. **No Rate Limiting:** Respect API limits
4. **Inefficient Caching:** Set appropriate TTLs
5. **Poor Batch Sizes:** Test optimal batch sizes
6. **No Monitoring:** Add metrics from day one
7. **Weak Testing:** Test edge cases thoroughly
8. **No Documentation:** Document while building

## Template Scripts

### Quick Start Script
```bash
#!/bin/bash
# setup_[api]_integration.sh

echo "Setting up [API] integration..."

# Create directories
mkdir -p data/api_responses
mkdir -p scripts/tests
mkdir -p docs/apis

# Copy templates
cp templates/api_client.py src/connections/[api]_client.py
cp templates/test_api.py scripts/test_[api]_api.py

# Run initial test
python scripts/test_[api]_api.py

echo "Setup complete. Edit configuration in config/apis/[api].yaml"
```

## Next Steps After Implementation

1. **Monitor for 24 hours:** Watch for issues
2. **Optimize caching:** Adjust TTLs based on usage
3. **Tune scheduling:** Optimize intervals
4. **Scale testing:** Add more resources
5. **Performance tuning:** Optimize slow queries
6. **Documentation review:** Update based on learnings