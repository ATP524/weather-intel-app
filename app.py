"""
Weather Intelligence — Databricks App (Flask REST API)

Pipeline this service exposes:
    NWS API  --/weather/sync-->  weather_documents (raw narrative text)
                                        |
                          (ingest notebook: chunk + embed)
                                        v
                        weather_embeddings / weather_chunk_embeddings
                                        |
                                 --/weather/search-->  ranked results (pgvector cosine)

Two endpoints:
    POST /weather/sync    Harvest active alerts + multi-day forecasts for a set
                          of locations from api.weather.gov and upsert the
                          normalized narrative text into Lakebase.
    POST /weather/search  Embed a natural-language query and return the most
                          semantically similar weather documents via pgvector's
                          `<=>` cosine operator.

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import json
import logging
import os

import requests
from flask import Flask, jsonify, render_template, request
from psycopg2.extras import execute_values

import lakebase
from weather_client import (
    WeatherClient,
    air_quality,
    daily_weather,
    geocode,
    resolve_location,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

# Table names are env-overridable so the same code can target different schemas
# (e.g. a scratch table in dev) without edits. Defaults match the sql/ DDL.
DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
CHUNK_EMBEDDINGS_TABLE = os.environ.get(
    "WEATHER_CHUNK_EMBEDDINGS_TABLE", "weather_chunk_embeddings"
)

# Default locations to harvest when the caller doesn't specify any. Comma-
# separated env var, each entry a curated city or a "lat,lon" pair.
DEFAULT_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get(
        "WEATHER_LOCATIONS", "Chicago, IL;Austin, TX;Miami, FL;Oklahoma City, OK"
    ).split(";")
    if loc.strip()
]

# The embedding model MUST match the one used by the ingest notebook, or query
# vectors live in a different space than the stored vectors and similarity is
# meaningless. 384-dim all-MiniLM-L6-v2 pairs with the vector(384) columns.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Sliding-window chunking for embedding — identical to the notebook
# (notebooks/ingest_weather_embeddings.py) so both write paths stay consistent.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# top_k guardrails: clamp caller-supplied values into a sane range so a typo
# like top_k=100000 can't ask Postgres to materialize the whole table.
_MIN_TOP_K = 1
_MAX_TOP_K = 20

# --- Lazy embedding model --------------------------------------------------
# Loading a sentence-transformers model is slow (downloads weights, warms up
# torch). We load it once, lazily, on first search — not at import time — so the
# app boots fast and /weather/sync (which needs no model) never pays the cost.
_embedding_model = None


def get_embedding_model():
    """Load and cache the sentence-transformers model on first use."""
    global _embedding_model
    if _embedding_model is None:
        # Imported lazily too: keeps `python app.py` startup and /healthz cheap
        # even in images where torch is heavy.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def ensure_weather_tables():
    """
    Ensure the weather schema + document table exist.

    If an operator already provisioned the schema (by running the sql/ DDL
    scripts), the app's DB role usually does NOT own those objects — and
    CREATE INDEX / ALTER require ownership, which is what raised
    "must be owner of table weather_documents" on /weather/sync. So we first
    check whether the table exists; if it does, we assume the operator manages
    the schema and skip all DDL, doing DML only. The CREATE statements below run
    only on a genuinely fresh database (e.g. local dev), where the app creates —
    and therefore owns — the table.
    """
    present = lakebase.run_query(
        "SELECT to_regclass(%s) IS NOT NULL AS present",
        (f"weather.{DOCUMENTS_TABLE}",),
    )[0]["present"]
    if present:
        return

    lakebase.run_write("CREATE SCHEMA IF NOT EXISTS weather")
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
            id             TEXT PRIMARY KEY,
            location       TEXT NOT NULL,
            source_type    TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
            headline       TEXT,
            event          TEXT,
            narrative_text TEXT NOT NULL,
            effective_at   TIMESTAMPTZ,
            expires_at     TIMESTAMPTZ,
            payload        JSONB NOT NULL,
            synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_source_type "
        f"ON {DOCUMENTS_TABLE} (source_type)"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Return JSON (never an HTML error page) for any unhandled error, so a
    client calling resp.json() never chokes on an HTML 500 page."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to sync locations and run semantic weather search."""
    return render_template("index.html")


# ===========================================================================
# Part 1 — Harvest:  POST /weather/sync
# ===========================================================================
@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Harvest active alerts + multi-day forecasts for each requested location and
    upsert the normalized narrative documents into weather_documents.

    Body (optional JSON):
        {"locations": ["Chicago, IL", "30.27,-97.74"], "limit": 50}

    - locations: curated city names and/or "lat,lon" pairs. Defaults to
      WEATHER_LOCATIONS when omitted.
    - limit: max documents to write PER LOCATION (protects against a state with
      dozens of active alerts flooding the table in one call).

    Then vectorizes the freshly-synced docs inline so each city is searchable
    immediately (near-real-time), instead of waiting on a separate batch run.

    Returns: {"synced": N, "embedded": D, "chunks": C, "locations": [...], "skipped": [...]}
    """
    ensure_weather_tables()
    client = WeatherClient()

    body = request.json if request.is_json else {}
    locations = body.get("locations") or DEFAULT_LOCATIONS
    limit = _coerce_int(body.get("limit"), default=20, low=1, high=500)

    total = 0
    synced_locations = []
    skipped = []
    embedded = []  # documents to vectorize after upsert

    for location in locations:
        # Resolve the location up front; a bad entry shouldn't abort the whole
        # batch, so we record it in `skipped` and continue.
        try:
            lat, lon, label = resolve_location(location)
            grid = client.resolve_point(lat, lon)
        except (ValueError, requests.HTTPError, KeyError) as exc:
            logger.warning("Skipping location %r: %s", location, exc)
            skipped.append({"location": location, "reason": str(exc)})
            continue

        # NWS reverse-geocodes coordinates to a city/state; prefer that as the
        # display label when the caller passed raw coordinates.
        if grid.get("city") and grid.get("state"):
            label = f"{grid['city']}, {grid['state']}"

        # Alerts are keyed by US state; forecasts by grid cell. Harvest both,
        # then upsert up to `limit` documents for this location.
        docs = []
        if grid.get("state"):
            docs.extend(client.get_active_alerts(grid["state"], label))
        docs.extend(
            client.get_forecast(grid["office"], grid["grid_x"], grid["grid_y"], label)
        )

        docs = docs[:limit]
        total += _upsert_documents(docs)
        embedded.extend(docs)
        synced_locations.append(label)

    # Vectorize the just-synced docs into the pgvector tables so each city is
    # searchable immediately — the near-real-time payoff of the psycopg2/Lakebase
    # design. Same execute_values + %s::vector path the notebook uses (Part 2),
    # run inline here rather than as a separate batch job. See _embed_and_store.
    n_docs, n_chunks = _embed_and_store(embedded)

    return jsonify({
        "synced": total,
        "embedded": n_docs,
        "chunks": n_chunks,
        "locations": synced_locations,
        "skipped": skipped,
    })


def _upsert_documents(docs: list[dict]) -> int:
    """Upsert normalized weather documents, keyed on the stable `id`.

    ON CONFLICT (id) DO UPDATE means re-running /weather/sync refreshes existing
    rows in place instead of duplicating — the deduplication the assignment asks
    for, achieved purely through deterministic ids from weather_client.py.
    """
    if not docs:
        return 0

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in docs:
                cur.execute(
                    f"""
                    INSERT INTO {DOCUMENTS_TABLE} (
                        id, location, source_type, headline, event,
                        narrative_text, effective_at, expires_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        location       = EXCLUDED.location,
                        source_type    = EXCLUDED.source_type,
                        headline       = EXCLUDED.headline,
                        event          = EXCLUDED.event,
                        narrative_text = EXCLUDED.narrative_text,
                        effective_at   = EXCLUDED.effective_at,
                        expires_at     = EXCLUDED.expires_at,
                        payload        = EXCLUDED.payload,
                        synced_at      = EXCLUDED.synced_at
                    """,
                    (
                        doc["id"],
                        doc["location"],
                        doc["source_type"],
                        doc.get("headline"),
                        doc.get("event"),
                        doc["narrative_text"],
                        doc.get("effective_at"),
                        doc.get("expires_at"),
                        # payload is raw NWS JSON; psycopg2 needs it serialized
                        # to a string for the JSONB column.
                        json.dumps(doc.get("payload", {})),
                    ),
                )
                count += 1
            conn.commit()
    return count


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding character window (mirrors the notebook's chunker). Short text
    returns a single chunk; long alert text splits with overlap to preserve
    context across boundaries."""
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def _vector_literal(vec) -> str:
    """Render an embedding as a pgvector text literal '[0.1,0.2,...]' for the
    explicit `%s::vector` cast (works without a registered pgvector adapter)."""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def _embed_and_store(docs: list[dict]) -> tuple[int, int]:
    """Embed the given documents and upsert vectors into both pgvector tables.

    Runs the SAME psycopg2 + execute_values + %s::vector path as the Part-2
    notebook, but inline in the sync request so a freshly-synced city is
    searchable immediately (the near-real-time payoff). Idempotent via ON
    CONFLICT, so re-syncing refreshes vectors in place.

    Returns (n_document_vectors, n_chunk_vectors).
    """
    if not docs:
        return (0, 0)

    model = get_embedding_model()  # same model used by /weather/search
    doc_rows, chunk_rows = [], []
    for doc in docs:
        narrative = doc["narrative_text"]
        # Document-level vector: headline + narrative as one unit.
        doc_input = f"{doc.get('headline') or ''}\n\n{narrative}".strip()
        doc_rows.append((
            doc["id"], doc["location"], doc.get("headline"), doc["source_type"],
            _vector_literal(model.encode(doc_input)), EMBEDDING_MODEL_NAME,
        ))
        # Chunk-level vectors: split the narrative, embed each window.
        chunks = _chunk(narrative)
        if chunks:
            for idx, (chunk, vec) in enumerate(zip(chunks, model.encode(chunks))):
                chunk_rows.append((
                    f"{doc['id']}:{idx}", doc["id"], idx, chunk,
                    _vector_literal(vec), EMBEDDING_MODEL_NAME,
                ))

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {EMBEDDINGS_TABLE}
                    (document_id, location, headline, source_type, embedding, model_name)
                VALUES %s
                ON CONFLICT (document_id) DO UPDATE SET
                    location    = EXCLUDED.location,
                    headline    = EXCLUDED.headline,
                    source_type = EXCLUDED.source_type,
                    embedding   = EXCLUDED.embedding,
                    model_name  = EXCLUDED.model_name,
                    created_at  = now()
                """,
                doc_rows,
                template="(%s, %s, %s, %s, %s::vector, %s)",
                page_size=200,
            )
            if chunk_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {CHUNK_EMBEDDINGS_TABLE}
                        (id, document_id, chunk_index, chunk_text, embedding, model_name)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        chunk_text  = EXCLUDED.chunk_text,
                        embedding   = EXCLUDED.embedding,
                        model_name  = EXCLUDED.model_name,
                        created_at  = now()
                    """,
                    chunk_rows,
                    template="(%s, %s, %s, %s, %s::vector, %s)",
                    page_size=200,
                )
        conn.commit()
    return (len(doc_rows), len(chunk_rows))


# ===========================================================================
# Location autocomplete:  GET /weather/geocode
# ===========================================================================
@app.route("/weather/geocode", methods=["GET"])
def geocode_locations():
    """
    Type-ahead helper for the UI. Resolves a partial place name to ranked US
    coordinate candidates so users never have to guess a format or know a
    curated list. The browser calls this instead of hitting the geocoder
    directly, which keeps the provider swappable and CORS a non-issue.

    Query params:
        q     — the (partial) place name; empty returns no results.
        limit — max candidates, clamped 1..10 (default 5).

    Returns: {"query": q, "results": [{"label","lat","lon","state"}, ...]}
    """
    q = request.args.get("q", "").strip()
    limit = _coerce_int(request.args.get("limit"), default=5, low=1, high=10)
    if not q:
        return jsonify({"query": q, "results": []})
    try:
        results = geocode(q, limit=limit)
    except requests.HTTPError as exc:
        logger.warning("Geocoding failed for %r: %s", q, exc)
        return jsonify({"error": f"Geocoding failed: {exc}"}), 502
    return jsonify({"query": q, "results": results})


# ===========================================================================
# Live conditions (UI panel):  GET /weather/conditions
# ===========================================================================
@app.route("/weather/conditions", methods=["GET"])
def weather_conditions():
    """
    Yesterday/today/tomorrow weather + current air quality for one location,
    powering the UI's left-hand Conditions panel.

    Sourced from Open-Meteo (free, no key) — NWS is forward-only (no yesterday)
    and has no air-quality data. This reads live and does NOT touch Lakebase, so
    it works before anything is synced. The graded search pipeline stays NWS.

    Query param: location — a US city name or "lat,lon".
    Returns: {"location", "days": [...3...], "air_quality": {...}}
    """
    location = request.args.get("location", "").strip()
    if not location:
        return jsonify({"error": "'location' is required"}), 400

    try:
        lat, lon, label = resolve_location(location)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        days = daily_weather(lat, lon)
        air = air_quality(lat, lon)
    except requests.HTTPError as exc:
        logger.warning("Conditions lookup failed for %r: %s", location, exc)
        return jsonify({"error": f"Conditions lookup failed: {exc}"}), 502

    return jsonify({"location": label, "days": days, "air_quality": air})


# ===========================================================================
# Part 3 — Retrieve:  POST /weather/search
# ===========================================================================
@app.route("/weather/search", methods=["POST"])
def search_weather():
    """
    Semantic search over ingested weather documents using pgvector cosine
    similarity.

    Body (JSON):
        {
            "query": "flash flood risk this weekend",
            "top_k": 5,                      # clamped to 1..20
            "search_mode": "chunk",          # "chunk" (default) | "document"
            "source_type": "alert",          # optional: "alert" | "forecast"
            "primary_city": "Milwaukee"      # optional: the user's active city
        }

    Each match carries location, headline, chunk_text (or narrative_text in
    document mode), source_type, effective/expires times, and cosine similarity
    in [0, 1]. When `primary_city` is given, results are split into that city's
    matches (`primary`) and everywhere else (`elsewhere`) — so the user's own
    location is always surfaced first, even on a weak match; otherwise a single
    `results` list is returned.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    body = request.json
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "'query' is required and must be non-empty"}), 400

    top_k = _coerce_int(body.get("top_k"), default=5, low=_MIN_TOP_K, high=_MAX_TOP_K)
    search_mode = body.get("search_mode", "chunk")
    if search_mode not in ("chunk", "document"):
        return jsonify({"error": "search_mode must be 'chunk' or 'document'"}), 400

    source_type = body.get("source_type")
    if source_type is not None and source_type not in ("alert", "forecast"):
        return jsonify({"error": "source_type must be 'alert' or 'forecast'"}), 400

    # Optional "primary" city = the user's active location. When present, we run
    # the search twice — filtered TO that city, and EXCLUDING it — so their own
    # location is always shown first (even a weak match), with other cities that
    # match the pattern surfaced separately as context ("Elsewhere").
    primary_city = (body.get("primary_city") or "").strip() or None
    if primary_city and (primary_city[0].isdigit() or primary_city[0] == "-"):
        primary_city = None  # a raw "lat,lon" label — can't anchor by city name

    # Embed the query with the SAME model used for ingestion, cast via %s::vector.
    model = get_embedding_model()
    query_vec = model.encode(query).tolist()
    search = _search_chunks if search_mode == "chunk" else _search_documents

    try:
        if primary_city:
            primary = search(query_vec, top_k, source_type, city=primary_city, city_include=True)
            elsewhere = search(query_vec, top_k, source_type, city=primary_city, city_include=False)
            return jsonify({
                "query": query,
                "search_mode": search_mode,
                "source_type": source_type,
                "primary_city": primary_city,
                "primary": primary,
                "elsewhere": elsewhere,
                "count": len(primary) + len(elsewhere),
            })
        results = search(query_vec, top_k, source_type)
    except Exception as exc:  # includes the "table doesn't exist yet" case
        logger.exception("Vector search failed")
        return jsonify({"error": f"Vector search failed: {exc}"}), 500

    return jsonify(
        {
            "query": query,
            "search_mode": search_mode,
            "source_type": source_type,
            "count": len(results),
            "results": results,
        }
    )


def _city_condition(city_include: bool) -> str:
    """Match on the city token of d.location (e.g. 'Milwaukee' from 'Milwaukee,
    WI'). NWS stores 'City, ST' while the picker uses 'City, State', so matching
    the full label would fail — the city token bridges the two."""
    op = "=" if city_include else "<>"
    return f"lower(split_part(d.location, ',', 1)) {op} lower(%s)"


def _search_chunks(query_vec: list[float], top_k: int, source_type: str | None,
                   city: str | None = None, city_include: bool = True):
    """Chunk-level cosine search: join chunk vectors back to their documents.

    Optional filters: `source_type`, and a `city` include/exclude (to split
    prioritized vs. "elsewhere" results). `1 - (embedding <=> query)` is cosine
    similarity in [0, 1]; the vector is passed twice (SELECT score + ORDER BY)
    because pgvector can't reuse the computed column in ORDER BY.
    """
    conds: list[str] = []
    params: list = [query_vec]
    if source_type:
        conds.append("d.source_type = %s"); params.append(source_type)
    if city:
        conds.append(_city_condition(city_include)); params.append(city)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    params += [query_vec, top_k]

    return lakebase.run_query(
        f"""
        SELECT
            d.id            AS document_id,
            d.location,
            d.headline,
            d.source_type,
            d.effective_at,
            d.expires_at,
            d.narrative_text,
            e.chunk_text,
            e.chunk_index,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {CHUNK_EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        {where}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )


def _search_documents(query_vec: list[float], top_k: int, source_type: str | None,
                      city: str | None = None, city_include: bool = True):
    """Document-level cosine search against the one-vector-per-document table."""
    conds: list[str] = []
    params: list = [query_vec]
    if source_type:
        conds.append("e.source_type = %s"); params.append(source_type)
    if city:
        conds.append(_city_condition(city_include)); params.append(city)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    params += [query_vec, top_k]

    return lakebase.run_query(
        f"""
        SELECT
            d.id            AS document_id,
            d.location,
            d.headline,
            d.source_type,
            d.effective_at,
            d.expires_at,
            d.narrative_text,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        {where}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )


def _coerce_int(value, default: int, low: int, high: int) -> int:
    """Parse `value` to an int and clamp to [low, high], falling back to default.

    Used for both `limit` (sync) and `top_k` (search) so malformed or
    out-of-range client input degrades gracefully instead of 500-ing.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)
