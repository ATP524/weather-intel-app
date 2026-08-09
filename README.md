# Weather Intelligence — Lakebase Vector Search App

A Databricks App that harvests **unstructured weather text** from the National
Weather Service API, embeds it into **Lakebase** (Databricks-managed Postgres +
`pgvector`), and exposes a **semantic search** REST API over it.

```
NWS API ──/weather/sync──▶ weather_documents ──embed──▶ pgvector ──/weather/search──▶ ranked results
```

> 📖 **Full design write-up, schema rationale, and end-to-end run guide:
> [`README_WEATHER.md`](README_WEATHER.md).** This page is the quick overview.

## What it does

- **Harvest** — `POST /weather/sync` pulls active alerts + multi-day forecasts
  for a set of locations from `api.weather.gov` and upserts the narrative text
  into Postgres. No API key required (NWS is open).
- **Embed** — a psycopg2 notebook chunks + embeds that text with
  `all-MiniLM-L6-v2` (384-dim) into two `pgvector` tables.
- **Retrieve** — `POST /weather/search` embeds a natural-language query and
  ranks documents by cosine similarity (`<=>`).

## File map

| File | Purpose |
|------|---------|
| `app.py` | Flask API: `/healthz`, `/` (UI), `POST /weather/sync`, `POST /weather/search` |
| `weather_client.py` | NWS API client (grid resolution, alerts, forecasts) + location resolver |
| `lakebase.py` | Lakebase connection helper (psycopg2, `search_path = weather`) |
| `sql/01–03_*.sql` | DDL for `weather_documents` + the two `pgvector` embedding tables |
| `notebooks/ingest_weather_embeddings.py` | Chunk + embed pipeline (psycopg2, no Spark JDBC) |
| `templates/index.html` | Minimal UI: sync form + semantic search |
| `setup_secrets.py` | One-time: store the Lakebase URL in a Databricks secret scope |
| `app.yaml` / `databricks.yml` | Databricks App deploy config / Asset Bundle |

## Quick start

```bash
# 1. Local env
cp .env.example .env            # fill in LAKEBASE_URL and WEATHER_USER_AGENT (a real contact)
pip install -r requirements.txt

# 2. Create tables (run in the Lakebase SQL editor, in order)
#    sql/01_setup_weather_documents.sql
#    sql/02_setup_weather_embeddings.sql
#    sql/03_setup_weather_chunk_embeddings.sql

# 3. Run the API
python app.py                   # http://localhost:8000

# 4. Harvest, then embed (Databricks notebook), then search
curl -sX POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Oklahoma City, OK"], "limit": 50}'
```

On Databricks: store the Lakebase secret with `python setup_secrets.py`, deploy
the app via `app.yaml`, and run `notebooks/ingest_weather_embeddings.py` to build
the vectors. See [`README_WEATHER.md`](README_WEATHER.md) for the full walkthrough.
