# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

A retrieval-augmented pipeline over free-text weather data:

```
                    POST /weather/sync  (harvest + embed in one call)
NWS API ──harvest──▶ weather_documents ──chunk + embed inline (psycopg2)──▶
                                            weather_embeddings + weather_chunk_embeddings
                                            (pgvector, 384-dim)
                                                      │
                          POST /weather/search ──▶ ranked results (cosine `<=>`),
                                                   your active city prioritized
```

`POST /weather/sync` harvests **and vectorizes** in a single call, so a city is
searchable immediately (the near-real-time payoff of psycopg2 + Lakebase — no
separate batch step). `POST /weather/search` then returns the most semantically
relevant documents by cosine similarity, with the user's active city surfaced
first. `notebooks/ingest_weather_embeddings.py` provides the same embedding as a
standalone batch job for bulk reprocessing (the Part-2 deliverable).

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

**Deliberate source split (NWS vs Open-Meteo):** the graded pipeline — harvest
→ embed → search — is **100% NWS**. The UI's left-hand **Live Conditions panel**
(`GET /weather/conditions`) is a display-only add-on that sources
yesterday/today/tomorrow summaries and current air quality (US AQI + PM2.5/10 +
ozone) from **Open-Meteo**, because NWS is forward-only (no "yesterday") and
carries no air-quality data. No allergen/pollen data is shown — it's empty for
the US on every free source (Open-Meteo's pollen is Europe-only; US pollen APIs
require paid keys).

**Interpreting search scores:** results are ranked by cosine similarity between
the query embedding and the alert/forecast text — i.e. *how closely an ingested
NWS document describes your query*, **not a probability the weather will occur**.
The UI buckets the raw cosine into four labelled tiers — **Strong ≥0.50**,
**Moderate 0.35–0.50**, **Weak 0.20–0.35**, **Faint <0.20** — as colored badges
(exact cosine in a hover tooltip), alongside each result's active/effective time
window. So "match" reads as "this is in the alerts/forecast for the next ~7
days," which is the app's real utility.

**Location-prioritized results:** when the user has an active city,
`POST /weather/search` returns two groups — `primary` (that city's best matches,
always shown first, even a weak one) and `elsewhere` (stronger matches in *other*
synced cities). This answers "is this pattern a risk where I am, and where else
is it happening?" — useful for travel. Matching keys on the city-name token of
`location` (NWS stores "City, ST" while the geocoder uses "City, State").

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

**Chunking:** `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (sliding character window),
applied identically by the app's embed-on-sync path (`app.py`) and the batch
notebook. Most forecast periods are short → a single chunk; combined alert
`description`+`instruction` is where chunking matters. Overlap preserves context
across boundaries.

**Indexing:** HNSW with `vector_cosine_ops` on both embedding columns — pairs
with the `<=>` cosine operator used in search. A mismatched opclass silently
disables the index.

---

## 3. Run the pipeline end-to-end

### Step 0 — Create the tables & grant access (once)
Run the three DDL scripts in the Lakebase SQL editor, in order:
`sql/01_setup_weather_documents.sql`, `sql/02_setup_weather_embeddings.sql`,
`sql/03_setup_weather_chunk_embeddings.sql` (see `sql/README.md`). Then grant the
app's role DML on the `weather` schema so it can write **documents *and*
embeddings** (embed-on-sync writes both):
```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA weather TO <app_role>;
ALTER DEFAULT PRIVILEGES IN SCHEMA weather GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO <app_role>;
```

### Step 1 — Harvest **and embed** (`POST /weather/sync`)
A single call fetches, normalizes, upserts into `weather_documents`, **and
vectorizes inline** (psycopg2 + `execute_values` + `%s::vector`) into both
embedding tables — so the city is searchable immediately, with no separate batch
step. Idempotent via `ON CONFLICT`, so re-syncing refreshes in place.
```bash
curl -sX POST <app-url>/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Oklahoma City, OK"], "limit": 20}'
```
Expected:
```json
{"synced": 28, "embedded": 28, "chunks": 34,
 "locations": ["Chicago, IL", "Oklahoma City, OK"], "skipped": []}
```

### Step 2 — Retrieve (`POST /weather/search`)
Pass `primary_city` to prioritize the user's active city; results split into
that city vs. everywhere else. Each match includes its `effective_at`/`expires_at`.
```bash
curl -sX POST <app-url>/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "tornado risk", "top_k": 5, "primary_city": "Milwaukee"}'
```
Expected shape:
```json
{
  "query": "tornado risk", "search_mode": "chunk", "source_type": null,
  "primary_city": "Milwaukee",
  "primary":   [{"location": "Milwaukee, WI", "headline": "This Afternoon",
                 "source_type": "forecast", "similarity": 0.19, "chunk_text": "..."}],
  "elsewhere": [{"location": "Oklahoma City, OK", "headline": "Tornado Watch",
                 "source_type": "alert", "similarity": 0.63, "chunk_text": "..."}],
  "count": 2
}
```
Omit `primary_city` to get a single corpus-wide `results` list instead.

**Search options:** `search_mode` = `chunk` (default; returns the matching
passage) or `document` (whole-doc); `top_k` clamped 1–20; `source_type` =
`alert` | `forecast`; `primary_city` anchors the prioritized split.

### (Optional) Batch embedding — `notebooks/ingest_weather_embeddings.py`
Embed-on-sync covers normal use. The notebook is the **Part-2 deliverable** and a
bulk/backfill tool: it reads `weather_documents` via psycopg2 and re-embeds into
the same tables (idempotent). Run it in Databricks only when you need to re-embed
everything — e.g. after changing the model or a large one-off import.

---

## 4. Running the app (Databricks App)

This runs as a **Databricks App** deployed from the repo — no local server
needed. In the Databricks UI:

1. Create/point an App at this repo; `app.yaml` provides the run command + env.
2. Set env: `LAKEBASE_SECRET_SCOPE=database-day2`, `WEATHER_USER_AGENT` (a real
   contact — NWS requires it), `EMBEDDING_MODEL`, and optionally `WEATHER_LOCATIONS`.
3. Add the `lakebase-url` **secret as an App resource** — this grants the app's
   service principal read access to it (see `setup_secrets.py`, which stores the
   base64-encoded Lakebase URL in the `database-day2` scope). NWS needs no key.
4. **Deploy** (and redeploy after any code change). The `/` route serves the UI.

> Local dev is optional: `cp .env.example .env`, `pip install -r requirements.txt`,
> `python app.py` (serves `:8000`) — but Databricks Apps is the deployment target.

---

## 5. Known limitations & next steps

- **US-only + external geocoder dependency.** City names resolve via Open-Meteo
  (filtered to US, since NWS only covers the US). Non-US places won't resolve,
  and syncing an un-cached city adds one geocoding round-trip. The in-process
  cache is per-worker and not shared across app replicas.
- **Synchronous embedding on sync.** `/weather/sync` embeds inline, so a large
  multi-city sync blocks the request for the encode time (fine for ~20 docs/city;
  the model loads on the first sync of a session). Embedding covers just the
  freshly-synced docs, keyed by stable id; use the batch notebook for heavy
  backfills.
- **Prioritized search matches by city name.** The `primary`/`elsewhere` split
  keys on the city-name token of `location` (NWS "City, ST" vs geocoder "City,
  State"); a raw `"lat,lon"` pick falls back to a corpus-wide search.
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
| NWS API client (+ geocode, conditions, air quality) | `weather_client.py` |
| REST endpoints | `app.py` — `POST /weather/sync`, `POST /weather/search`, `GET /weather/geocode`, `GET /weather/conditions` |
| DDL / migrations | `sql/01`–`03`, `lakebase.py` (search_path) |
| Embedding — psycopg2 | inline in `app.py` (`/weather/sync`, embed-on-sync) **and** batch `notebooks/ingest_weather_embeddings.py` |
| UI | `templates/index.html` (type-ahead location, live conditions, prioritized search) |
| This document | `README_WEATHER.md` |
