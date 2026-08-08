-- Setup script for weather_documents table
-- Run this manually in your Lakebase Postgres database before running the notebook

-- Create the weather documents table
CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,  -- City name: "Chicago, IL" or state code "FL"
    latitude DOUBLE PRECISION,  -- Latitude coordinate (NULL for state-wide alerts)
    longitude DOUBLE PRECISION, -- Longitude coordinate (NULL for state-wide alerts)
    source_type TEXT NOT NULL, -- 'alert', 'state_alert', 'forecast', 'hourly_forecast', or 'discussion'
    headline TEXT,
    event TEXT,
    narrative_text TEXT NOT NULL,
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_weather_documents_location 
ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type 
ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_effective_at 
ON weather_documents (effective_at);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
