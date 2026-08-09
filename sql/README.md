# SQL Setup — Weather Vector Search (Lakebase / Postgres + pgvector)

Run these three scripts **once, in order**, in your Lakebase SQL editor before
running the ingest notebook (`notebooks/ingest_weather_embeddings.py`). Each
script is idempotent (`CREATE ... IF NOT EXISTS`) and self-contained (it sets up
the `weather` schema and search_path itself), so re-running is safe.

## Setup order

| # | File | Creates | Purpose |
|---|------|---------|---------|
| 1 | `01_setup_weather_documents.sql` | `weather.weather_documents` | Raw NWS alerts + forecast periods (the text we embed) |
| 2 | `02_setup_weather_embeddings.sql` | `weather.weather_embeddings` | **Document-level** vectors — one per document |
| 3 | `03_setup_weather_chunk_embeddings.sql` | `weather.weather_chunk_embeddings` | **Chunk-level** vectors — many per document |

Run 1 first: the two embedding tables have a foreign key back to
`weather_documents`.

## Embedding dimension

All three scripts hardcode `vector(384)` for
`sentence-transformers/all-MiniLM-L6-v2`. The pgvector column width must match
the model exactly. If you swap models, update the dimension in **both** file 02
and file 03:

| Model | Dim |
|-------|-----|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `sentence-transformers/all-mpnet-base-v2` | 768 |
| `BAAI/bge-large-en-v1.5` | 1024 |

## Why there is no "cast arrays to vectors" step anymore

The original day-2 setup wrote embeddings with **Spark JDBC**, which can't write
pgvector's `VECTOR` type directly — it landed `DOUBLE PRECISION[]` arrays that
then needed a manual `UPDATE ... SET embedding = embedding::vector` pass.

This pipeline writes with **psycopg2** and casts in-SQL via `%s::vector` (see the
ingest notebook). Vectors land as real pgvector values on insert, so:

- ✅ No post-processing cast step
- ✅ `ON CONFLICT` upserts work (idempotent re-runs)
- ✅ HNSW indexes are usable immediately

## Quick end-to-end validation

After running the DDL, syncing, and embedding, confirm each layer is populated:

```sql
SET search_path TO weather, public;

SELECT 'documents'         AS layer, COUNT(*) FROM weather_documents
UNION ALL
SELECT 'doc_embeddings',            COUNT(*) FROM weather_embeddings
UNION ALL
SELECT 'chunk_embeddings',          COUNT(*) FROM weather_chunk_embeddings;
```

Expected: `documents` >= `doc_embeddings` (one vector per doc), and
`chunk_embeddings` >= `doc_embeddings` (documents with long text split into
multiple chunks).
