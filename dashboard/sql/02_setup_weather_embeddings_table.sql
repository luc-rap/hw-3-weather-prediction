-- Setup script for weather_embeddings table
-- Run this after weather_documents table is created
-- This table stores chunked embeddings - each weather document can have multiple chunks

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the weather embeddings table with chunking support
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,  -- Composite: document_id + '_' + chunk_index
    document_id TEXT NOT NULL,  -- FK to weather_documents.id
    chunk_index INTEGER NOT NULL,  -- 0-based chunk number within the document
    chunk_text TEXT NOT NULL,  -- The text chunk that was embedded
    embedding vector(384) NOT NULL,  -- Match your model dimension (384 for all-MiniLM-L6-v2)
    model_name TEXT NOT NULL,  -- Which embedding model was used
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique chunks per document
    UNIQUE(document_id, chunk_index)
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_vector 
ON weather_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Tune based on your data volume

-- Index for filtering by document
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id 
ON weather_embeddings(document_id);

-- Index for filtering by model
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_model 
ON weather_embeddings(model_name);
