-- Audit Trail: API Responses
CREATE TABLE api_response_audit (
    id BIGSERIAL PRIMARY KEY,
    response_id UUID NOT NULL DEFAULT gen_random_uuid(),
    endpoint VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    response_size_bytes INTEGER,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    rate_limit_remaining INTEGER,
    field_count INTEGER, -- Number of fields in response
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(response_id)
);

-- Indexes
CREATE INDEX idx_api_audit_endpoint ON api_response_audit(endpoint);
CREATE INDEX idx_api_audit_created ON api_response_audit(created_at);
