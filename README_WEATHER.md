# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

A retrieval-augmented pipeline over free-text weather data:

```
NWS API  ──POST /weather/sync──▶  weather_documents        (raw narrative text, Postgres)
                                        │
                    ingest_weather_embeddings.py  (chunk + embed, psycopg2)
                                        ▼
                    weather_embeddings  +  weather_chunk_embeddings   (pgvector, 384-dim)
                                        │
                                ──POST /weather/search──▶  ranked results (cosine `<=>`)
```

`POST /weather/search {"query": "flash flood risk this weekend"}` returns the
most semantically relevant weather documents, ranked by vector similarity.

---

## 1. Data source: National Weather Service (`api.weather.gov`) — and why

| Reason | Detail |
|--------|--------|
| **Free, no API key** | No auth plumbing; the work stays on harvest → vectorize → retrieve. |
| **Rich unstructured text** | Alerts carry free-text `description` + `instruction`; forecasts carry per-period `detailedForecast` narratives — ideal embedding inputs. |
| **Two complementary sources** | We harvest **both** alerts and forecasts and tag each row with `source_type`, so retrieval can span or filter by kind. |

**One non-obvious NWS constraint:** every request must send a descriptive
`User-Agent` with a contact, or NWS returns **HTTP 403**. It is set via
`WEATHER_USER_AGENT` (see `.env.example`). There is no `Authorization` header —
this API is open. Also, `/points` coordinates must be ≤ 4 decimal places (NWS
301-redirects over-precise coords); the client rounds automatically.

**NWS grid model (why sync is a 2-step call):** coordinates don't yield a
forecast directly. `GET /points/{lat},{lon}` returns a forecast **office** +
`gridX/gridY`, which then feed `GET /gridpoints/{office}/{x},{y}/forecast`.
Alerts are queried by **state** (`GET /alerts/active?area={ST}`). `/points` also
reverse-geocodes to a city/state, which we use as the display label.

**Location input:** `/weather/sync` accepts **any US city name** or a raw
`"lat,lon"` pair. Because NWS itself does no geocoding, `resolve_location()`
resolves names in four tiers (cheapest first): a small curated `CITY_COORDS`
seed map (offline), raw coordinate parsing, an in-process cache, then the
**Open-Meteo geocoding API** (free, no key) for everything else — filtered to
US results since that's NWS's coverage. The UI's location box is a type-ahead
backed by `GET /weather/geocode`, so users pick a real match (disambiguating,
e.g., Milwaukee, WI vs Milwaukie, OR) instead of guessing a format.

---

## 2. Schema decisions

Three tables in a dedicated `weather` schema (`sql/01`–`sql/03`). `lakebase.py`
sets `search_path = weather, public`.

### `weather_documents` — raw store
| Column | Notes |
|--------|-------|
| `id` (PK) | Alert URN (stable) or `sha256(location|period_start)` for forecasts → deterministic → dedup on re-sync |
| `location`, `source_type` (`alert`/`forecast`), `headline`, `event` | `source_type` has a `CHECK` and powers the optional search filter |
| `narrative_text` | The free text embedded (alert `description`+`instruction`, or `detailedForecast`) |
| `effective_at`, `expires_at`, `payload` (JSONB), `synced_at` | `payload` keeps the raw NWS feature for provenance |

### `weather_embeddings` — document-level vectors
One row per document (`document_id` PK), embedding of `headline + narrative`.
Good for "most relevant documents overall."

### `weather_chunk_embeddings` — chunk-level vectors
The homework's required shape: `id`, `document_id` (FK), `chunk_index`,
`chunk_text`, `embedding vector(384)`, `model_name`, `created_at`. Long
narratives are split so retrieval can surface the exact matching passage.

**Embedding model / dimensions:** `sentence-transformers/all-MiniLM-L6-v2`,
**384-dim**, matching the day-2 news pipeline so both use the same
`vector_cosine_ops` / `<=>` conventions. The column width `vector(384)` must
equal the model's output; swap both together (see `sql/README.md`).

**Chunking:** `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (sliding character window).
Most forecast periods are short → a single chunk; combined alert
`description`+`instruction` is where chunking matters. Overlap preserves context
across boundaries.

**Indexing:** HNSW with `vector_cosine_ops` on both embedding columns — pairs
with the `<=>` cosine operator used in search. A mismatched opclass silently
disables the index.

---

## 3. Run the pipeline end-to-end

### Step 0 — Create the tables (once)
Run in the Lakebase SQL editor, in order:
`sql/01_setup_weather_documents.sql`, `sql/02_setup_weather_embeddings.sql`,
`sql/03_setup_weather_chunk_embeddings.sql`. See `sql/README.md`.

### Step 1 — Harvest (`POST /weather/sync`)
```bash
curl -sX POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Oklahoma City, OK"], "limit": 50}'
```
Expected:
```json
{"synced": 32, "locations": ["Chicago, IL", "Oklahoma City, OK"], "skipped": []}
```

### Step 2 — Embed (`notebooks/ingest_weather_embeddings.py`)
Run the notebook in Databricks (it reuses the `database-day2/lakebase-url`
secret). Its final cells print row counts and a retrieval smoke test. Expected:
`doc_embeddings == documents`, `chunk_embeddings >= documents`.

### Step 3 — Retrieve (`POST /weather/search`)
```bash
curl -sX POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "excessive heat warning", "top_k": 5, "search_mode": "chunk"}'
```
Expected shape:
```json
{
  "query": "excessive heat warning",
  "search_mode": "chunk",
  "source_type": null,
  "count": 5,
  "results": [
    {"location": "Oklahoma City, OK", "headline": "Heat Advisory",
     "source_type": "alert", "chunk_text": "* WHAT...Heat index values up to 106...",
     "chunk_index": 0, "similarity": 0.71}
  ]
}
```

**Search options:** `search_mode` = `chunk` (default, returns the matching
passage) or `document` (whole-doc scores); `top_k` clamped 1–20; optional
`source_type` = `alert` | `forecast`.

---

## 4. Local development

```bash
cp .env.example .env      # fill in LAKEBASE_URL and WEATHER_USER_AGENT
pip install -r requirements.txt
python app.py             # serves on :8000
```

---

## 5. Known limitations & next steps

- **US-only + external geocoder dependency.** City names resolve via Open-Meteo
  (filtered to US, since NWS only covers the US). Non-US places won't resolve,
  and syncing an un-cached city adds one geocoding round-trip. The in-process
  cache is per-worker and not shared across app replicas.
- **Full re-embed each run.** The notebook embeds every document each run
  (idempotent via upsert). For scale, add a LEFT JOIN anti-filter against
  `weather_embeddings` to embed only new rows.
- **Alert volatility.** NWS alerts expire; a scheduled re-sync (Databricks Job)
  would keep the corpus current. `expires_at` is stored to support pruning.
- **No LLM summary yet.** The stretch-goal RAG summary (`GET /weather/search`
  returning a natural-language synthesis of top hits) is not implemented.
- **HNSW is approximate.** Fast but not guaranteed exact nearest-neighbor; for
  small corpora an exact scan (or `ivfflat`) may return marginally different
  ordering. Benchmarking with/without the index is a listed stretch goal.

---

## Deliverables map

| Deliverable | File |
|-------------|------|
| NWS API client | `weather_client.py` |
| REST endpoints | `app.py` (`POST /weather/sync`, `POST /weather/search`) |
| DDL / migrations | `sql/01`–`03`, `lakebase.py` (search_path) |
| Embedding ingestion (psycopg2) | `notebooks/ingest_weather_embeddings.py` |
| This document | `README_WEATHER.md` |
