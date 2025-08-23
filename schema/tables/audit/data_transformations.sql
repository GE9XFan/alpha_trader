-- Audit Trail: Data Transformations
-- Tracks EVERY transformation from API to database
CREATE TABLE data_transformations (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    original_value TEXT,
    transformed_value TEXT,
    transformation_type VARCHAR(50) NOT NULL,
    api_response_id UUID NOT NULL,
    field_path TEXT, -- Original field path in API response
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_transform_table ON data_transformations(table_name);
CREATE INDEX idx_transform_response ON data_transformations(api_response_id);
CREATE INDEX idx_transform_type ON data_transformations(transformation_type);
CREATE INDEX idx_transform_field ON data_transformations(field_path);
