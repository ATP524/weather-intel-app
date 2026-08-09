-- ============================================================================
-- 01 - weather_documents  (raw, unstructured NWS text — the "document store")
-- ============================================================================
-- Run order: this file FIRST (the two embedding tables below FK back to it).
-- Run this manually in your Lakebase SQL editor before the ingest notebook.
--
-- This is the raw layer of the pipeline: one row per harvested NWS item
-- (an active alert OR a single forecast period). The ingest notebook reads
-- narrative_text from here, chunks + embeds it, and writes vectors into the
-- weather_embeddings / weather_chunk_embeddings tables.
--
-- Schema design notes:
--   * All app tables live in a dedicated `weather` schema (mirrors how the
--     day-1/day-2 app isolated everything under `weather`). lakebase.py
--     sets search_path to `weather, public`, so the Flask app sees these
--     tables unqualified.
--   * `source_type` discriminates the two harvest sources so /weather/search
--     can optionally filter to just alerts or just forecasts (stretch goal).
--   * `payload JSONB` keeps the full raw NWS feature for provenance/debugging
--     without forcing us to model every NWS field as a column.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS weather;
SET search_path TO weather, public;

CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,          -- alert URN, or sha256(location|period_start) for forecasts
    location       TEXT NOT NULL,             -- "Chicago, IL" or "41.88,-87.63"
    source_type    TEXT NOT NULL              -- 'alert' | 'forecast'
                   CHECK (source_type IN ('alert', 'forecast')),
    headline       TEXT,                      -- alert headline/event, or forecast period name ("Tonight")
    event          TEXT,                      -- alert event ("Flash Flood Warning"); NULL for forecasts
    narrative_text TEXT NOT NULL,             -- the free text we embed (description+instruction, or detailedForecast)
    effective_at   TIMESTAMPTZ,               -- alert effective time, or forecast period start
    expires_at     TIMESTAMPTZ,               -- alert expiry, or forecast period end
    payload        JSONB NOT NULL,            -- raw NWS feature/period, for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Filter index: /weather/search may narrow by source_type, and the ingest
-- notebook selects rows to embed. Both benefit from an index here.
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

-- ---------------------------------------------------------------------------
-- Validation: confirm the shape after running the DDL above.
-- ---------------------------------------------------------------------------
-- SELECT column_name, data_type
-- FROM information_schema.columns
-- WHERE table_schema = 'weather' AND table_name = 'weather_documents'
-- ORDER BY ordinal_position;
--
-- After /weather/sync runs, sanity-check the harvest:
-- SELECT source_type, COUNT(*) FROM weather_documents GROUP BY source_type;



-- ---------------------------------------------------------------------------
-- Permissions: lakebase being set to current_user email vs. app student 
-- reconciliation to avoid permission issues
-- ---------------------------------------------------------------------------
-- ALTER TABLE weather_documents REPLICA IDENTITY FULL;

-- cannot touch any object without schema usage
GRANT USAGE ON SCHEMA weather TO student;

-- the DML the app requires to sync/search
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES    IN SCHEMA weather TO student;

-- this particular app has no sequences (all PKs are TEXT, no SERIAL) harmless to run 
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA weather TO student;

-- future-proof applies to tables the 'running' roles creates later
ALTER DEFAULT PRIVILEGES IN SCHEMA weather
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO student;

-- similar to usage, no harm to run
ALTER DEFAULT PRIVILEGES IN SCHEMA weather
  GRANT USAGE, SELECT ON SEQUENCES TO student;


-- ---------------------------------------------------------------------------
-- If needed: run these in lakebase to add new columns
-- ---------------------------------------------------------------------------

-- ALTER TABLE weather.weather_documents ADD COLUMN IF NOT EXISTS X NUMERIC;
-- ALTER TABLE weather.weather_documents ADD COLUMN IF NOT EXISTS X NUMERIC;
-- ALTER TABLE weather.weather_documents ADD COLUMN IF NOT EXISTS X NUMERIC;
-- ALTER TABLE weather.weather_documents ADD COLUMN IF NOT EXISTS X TEXT;
-- ALTER TABLE weather.weather_documents ADD COLUMN IF NOT EXISTS X TEXT;
