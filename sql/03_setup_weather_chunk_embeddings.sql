-- ============================================================================
-- 03 - weather_chunk_embeddings  (CHUNK-level vectors — many rows per document)
-- ============================================================================
-- Run AFTER 01_setup_weather_documents.sql (this table FKs back to it).
--
-- This is the "fine-grained" retrieval table and the one that matches the
-- homework's Part-2 column contract (document_id, chunk_index, chunk_text,
-- embedding). Long narrative_text (e.g. a full alert = description +
-- instruction) is split into overlapping windows; each window ("chunk") gets
-- its own embedding. Retrieval can then surface the *specific passage* that
-- matched, not just the whole document.
--
-- For most NWS forecast periods the text is short enough to be a single chunk,
-- so many documents will have exactly one row here. Alerts are where chunking
-- earns its keep.
--
-- Dimensionality: vector(384) for all-MiniLM-L6-v2. Keep in sync with file 02.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS weather;
SET search_path TO weather, public;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_chunk_embeddings (
    id           TEXT PRIMARY KEY,            -- typically "<document_id>:<chunk_index>"
    document_id  TEXT NOT NULL
                 REFERENCES weather_documents (id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,               -- 0-based position of this chunk within the document
    chunk_text   TEXT NOT NULL,               -- the exact passage that was embedded (returned in search)
    embedding    VECTOR(384) NOT NULL,
    model_name   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Re-embedding a document should overwrite its chunks, not duplicate them.
    -- The ingest notebook upserts on this pair.
    UNIQUE (document_id, chunk_index)
);

-- HNSW cosine index — the workhorse for the default (chunk-mode) /weather/search.
CREATE INDEX IF NOT EXISTS idx_weather_chunk_embeddings_embedding
    ON weather_chunk_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- Speeds up the JOIN back to weather_documents for result metadata.
CREATE INDEX IF NOT EXISTS idx_weather_chunk_embeddings_document_id
    ON weather_chunk_embeddings (document_id);

-- ---------------------------------------------------------------------------
-- Validation
-- ---------------------------------------------------------------------------
-- SELECT column_name, data_type, udt_name
-- FROM information_schema.columns
-- WHERE table_schema = 'weather' AND table_name = 'weather_chunk_embeddings'
-- ORDER BY ordinal_position;
--
-- After the ingest notebook runs, see how chunking behaved:
-- SELECT document_id, COUNT(*) AS n_chunks
-- FROM weather_chunk_embeddings
-- GROUP BY document_id
-- ORDER BY n_chunks DESC
-- LIMIT 10;
