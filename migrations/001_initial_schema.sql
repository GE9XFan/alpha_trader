-- AlphaTrader Initial Database Schema
-- Foundation tables for system operation
-- Created: Day 2 of implementation

-- System logs table for structured logging
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    correlation_id UUID,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) NOT NULL,
    module VARCHAR(100) NOT NULL,
    function_name VARCHAR(100),
    message TEXT,
    metadata JSONB,
    environment VARCHAR(50),
    
    -- Indexes for performance
    INDEX idx_system_logs_correlation_id (correlation_id),
    INDEX idx_system_logs_timestamp (timestamp DESC),
    INDEX idx_system_logs_level (level),
    INDEX idx_system_logs_module (module)
);

-- API calls tracking table
CREATE TABLE IF NOT EXISTS api_calls (
    id SERIAL PRIMARY KEY,
    correlation_id UUID,
    api_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    status_code INTEGER,
    success BOOLEAN,
    error_message TEXT,
    request_params JSONB,
    response_data JSONB,
    
    -- Indexes for performance
    INDEX idx_api_calls_correlation_id (correlation_id),
    INDEX idx_api_calls_timestamp (timestamp DESC),
    INDEX idx_api_calls_api_name (api_name),
    INDEX idx_api_calls_endpoint (endpoint),
    INDEX idx_api_calls_success (success)
);

-- Health checks history
CREATE TABLE IF NOT EXISTS health_checks (
    id SERIAL PRIMARY KEY,
    check_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    details JSONB,
    error_message TEXT,
    
    -- Indexes for performance
    INDEX idx_health_checks_timestamp (check_time DESC),
    INDEX idx_health_checks_component (component),
    INDEX idx_health_checks_status (status)
);

-- Rate limiting tracking
CREATE TABLE IF NOT EXISTS rate_limit_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resource_name VARCHAR(100) NOT NULL,
    tokens_available INTEGER,
    tokens_requested INTEGER,
    request_allowed BOOLEAN,
    correlation_id UUID,
    
    -- Indexes for performance
    INDEX idx_rate_limit_timestamp (timestamp DESC),
    INDEX idx_rate_limit_resource (resource_name),
    INDEX idx_rate_limit_allowed (request_allowed)
);

-- Circuit breaker events
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    service_name VARCHAR(100) NOT NULL,
    event_type VARCHAR(20) NOT NULL, -- 'opened', 'closed', 'half-open', 'failure'
    failure_count INTEGER,
    recovery_timeout INTEGER,
    error_message TEXT,
    
    -- Indexes for performance
    INDEX idx_circuit_breaker_timestamp (timestamp DESC),
    INDEX idx_circuit_breaker_service (service_name),
    INDEX idx_circuit_breaker_event (event_type)
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metric_type VARCHAR(50) NOT NULL, -- 'db_query', 'cache_operation', 'api_call'
    operation_name VARCHAR(100) NOT NULL,
    duration_ms DECIMAL(10, 3),
    success BOOLEAN,
    metadata JSONB,
    
    -- Indexes for performance
    INDEX idx_performance_timestamp (timestamp DESC),
    INDEX idx_performance_type (metric_type),
    INDEX idx_performance_operation (operation_name),
    INDEX idx_performance_duration (duration_ms)
);

-- Audit log for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    correlation_id UUID,
    action VARCHAR(100) NOT NULL,
    user_identifier VARCHAR(100),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    result VARCHAR(20) NOT NULL, -- 'success', 'failure', 'error'
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    
    -- Indexes for performance
    INDEX idx_audit_timestamp (timestamp DESC),
    INDEX idx_audit_correlation_id (correlation_id),
    INDEX idx_audit_action (action),
    INDEX idx_audit_user (user_identifier),
    INDEX idx_audit_result (result)
);

-- System configuration history (for tracking config changes)
CREATE TABLE IF NOT EXISTS configuration_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    config_key VARCHAR(200) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by VARCHAR(100),
    change_reason TEXT,
    
    -- Indexes for performance
    INDEX idx_config_history_timestamp (timestamp DESC),
    INDEX idx_config_history_key (config_key)
);

-- Create a view for recent API performance
CREATE OR REPLACE VIEW recent_api_performance AS
SELECT 
    api_name,
    endpoint,
    COUNT(*) as call_count,
    AVG(response_time_ms) as avg_response_time_ms,
    MAX(response_time_ms) as max_response_time_ms,
    MIN(response_time_ms) as min_response_time_ms,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100 as success_rate,
    DATE_TRUNC('hour', timestamp) as hour
FROM api_calls
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY api_name, endpoint, DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC, api_name, endpoint;

-- Create a view for system health summary
CREATE OR REPLACE VIEW system_health_summary AS
SELECT 
    component,
    status,
    COUNT(*) as check_count,
    AVG(response_time_ms) as avg_response_time_ms,
    MAX(check_time) as last_check_time
FROM health_checks
WHERE check_time > NOW() - INTERVAL '1 hour'
GROUP BY component, status
ORDER BY component, status;

-- Grant permissions (adjust user as needed)
-- These will be configured based on environment
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alphatrader_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alphatrader_user;

-- Add comments for documentation
COMMENT ON TABLE system_logs IS 'Centralized logging table for all system events';
COMMENT ON TABLE api_calls IS 'Tracks all external API calls for monitoring and debugging';
COMMENT ON TABLE health_checks IS 'History of health check results for all components';
COMMENT ON TABLE rate_limit_events IS 'Tracks rate limiting events for capacity planning';
COMMENT ON TABLE circuit_breaker_events IS 'Circuit breaker state changes for reliability monitoring';
COMMENT ON TABLE performance_metrics IS 'Performance metrics for system optimization';
COMMENT ON TABLE audit_log IS 'Audit trail for compliance and security';
COMMENT ON TABLE configuration_history IS 'Tracks configuration changes over time';