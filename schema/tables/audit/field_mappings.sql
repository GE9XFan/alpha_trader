-- Field Mapping Verification
-- Ensures all 8,227 fields are mapped
CREATE TABLE field_mapping_verification (
    id BIGSERIAL PRIMARY KEY,
    field_name TEXT NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    field_type VARCHAR(50) NOT NULL,
    occurrences INTEGER,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(field_name, table_name, column_name)
);

-- Indexes
CREATE INDEX idx_field_map_field ON field_mapping_verification(field_name);
CREATE INDEX idx_field_map_table ON field_mapping_verification(table_name);
CREATE INDEX idx_field_map_verified ON field_mapping_verification(verified);
