-- ============================================================================
-- 02 - weather_embeddings  (DOCUMENT-level vectors — one row per document)
-- ============================================================================
-- Run AFTER 01_setup_weather_documents.sql (this table FKs back to it).
--
-- This is the "coarse" retrieval table: each weather_documents row gets ONE
-- embedding computed over its (headline + narrative_text). Good for
-- document-level search ("show me the most relevant alerts/forecasts overall").
-- The companion 03_ table stores finer-grained per-chunk vectors.
--
-- Dimensionality:
--   embedding is vector(384) because we use sentence-transformers/all-MiniLM-L6-v2,
--   which emits 384-dim vectors. The pgvector column width MUST match the model
--   exactly or inserts fail. If you swap models, change 384 here AND in file 03.
--     all-MiniLM-L6-v2 -> 384 | all-mpnet-base-v2 -> 768 | bge-large -> 1024
--
-- Why psycopg2 (not Spark JDBC): the ingest notebook casts Python lists to the
-- VECTOR type in-SQL via `%s::vector`, so vectors land as real pgvector values
-- immediately. That's why—unlike the original day-2 setup—there is NO separate
-- "cast arrays to vectors" post-processing step here.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS weather;
SET search_path TO weather, public;

-- pgvector is pre-enabled in this Lakebase instance; harmless if already present.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    -- One embedding per document, so the document id doubles as the PK.
    document_id  TEXT PRIMARY KEY
                 REFERENCES weather_documents (id) ON DELETE CASCADE,
    location     TEXT,                        -- denormalized for cheap result display
    headline     TEXT,                        -- denormalized for cheap result display
    source_type  TEXT,                        -- enables source_type-filtered search
    embedding    VECTOR(384) NOT NULL,
    model_name   TEXT NOT NULL,               -- provenance: which model produced this vector
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index with cosine ops. This is what makes the `<=>` ORDER BY in
-- /weather/search fast: without it, Postgres sequentially scans every vector.
-- vector_cosine_ops pairs with the `<=>` (cosine distance) operator; using a
-- mismatched opclass silently disables the index for that query.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- Optional: lets a source_type-filtered search still use an index on the filter.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_source_type
    ON weather_embeddings (source_type);

-- ---------------------------------------------------------------------------
-- Validation: confirm the VECTOR column really is vector(384), not an array.
-- ---------------------------------------------------------------------------
-- SELECT column_name, data_type, udt_name        -- udt_name should read 'vector'
-- FROM information_schema.columns
-- WHERE table_schema = 'weather' AND table_name = 'weather_embeddings'
-- ORDER BY ordinal_position;
--
-- After the ingest notebook runs:
-- SELECT COUNT(*) AS docs_embedded FROM weather_embeddings;
