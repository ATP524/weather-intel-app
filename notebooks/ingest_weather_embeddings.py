# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC Part 2 of the Weather Intelligence pipeline. This notebook reads raw NWS
# MAGIC narrative text from `weather_documents` (populated by the Flask app's
# MAGIC `POST /weather/sync`), embeds it with `sentence-transformers`, and writes
# MAGIC the vectors into two pgvector tables:
# MAGIC
# MAGIC | Table | Grain | Row count |
# MAGIC |-------|-------|-----------|
# MAGIC | `weather_embeddings` | one vector per document (headline + narrative) | = # documents |
# MAGIC | `weather_chunk_embeddings` | one vector per text chunk | >= # documents |
# MAGIC
# MAGIC ### Deliberately psycopg2, NOT Spark JDBC
# MAGIC Spark's JDBC writer cannot write pgvector's `VECTOR` type (it lands
# MAGIC `DOUBLE PRECISION[]` arrays) and cannot `ON CONFLICT`-upsert. We use
# MAGIC `psycopg2.extras.execute_values` and cast in-SQL with `%s::vector`, so
# MAGIC vectors are correct on insert and re-runs are idempotent. There is no
# MAGIC "cast arrays to vectors" cleanup step.
# MAGIC
# MAGIC Re-uses the SAME Lakebase secret (`database-day2` / `lakebase-url`) that
# MAGIC `lakebase.py` uses — no extra secrets needed.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Widgets let a scheduled Job override table names / model / chunk sizes
# MAGIC without editing the notebook.

# COMMAND ----------

# DBTITLE 1,Restart Python kernel to clear psycopg2 conflict
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("documents_table", "weather_documents", "Source table (raw docs)")
dbutils.widgets.text("embeddings_table", "weather_embeddings", "Dest table (doc vectors)")
dbutils.widgets.text("chunk_embeddings_table", "weather_chunk_embeddings", "Dest table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("lakebase_secret_scope", "database-day2", "Lakebase secret scope")
dbutils.widgets.text("lakebase_secret_key", "lakebase-url", "Lakebase secret key")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")
dbutils.widgets.text("batch_size", "200", "Docs to process per DB round-trip")

DOCUMENTS_TABLE = dbutils.widgets.get("documents_table")
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table")
CHUNK_EMBEDDINGS_TABLE = dbutils.widgets.get("chunk_embeddings_table")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
LAKEBASE_SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
LAKEBASE_SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

# The pgvector column width must equal the model's output dim. Switch on the
# model so swapping the widget also tells us the expected dimension (we assert
# it against the real vector length after loading the model, below).
_MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
}
EMBEDDING_DIM = _MODEL_DIMS.get(EMBEDDING_MODEL_NAME)
if EMBEDDING_DIM is None:
    raise ValueError(
        f"Unknown model {EMBEDDING_MODEL_NAME!r}; add its dim to _MODEL_DIMS "
        "and make sure the vector(N) columns in sql/ match."
    )
print(f"Model {EMBEDDING_MODEL_NAME!r} -> expecting {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakebase connection (same secret/decoding scheme as `lakebase.py`)

# COMMAND ----------

# DBTITLE 1,Parse connection URL from the Databricks secret
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

_secret = w.secrets.get_secret(scope=LAKEBASE_SECRET_SCOPE, key=LAKEBASE_SECRET_KEY)
_lakebase_url = base64.b64decode(_secret.value).decode("utf-8")
_parsed = urlparse(_lakebase_url)

DB = dict(
    host=_parsed.hostname,
    port=_parsed.port or 5432,
    dbname=_parsed.path.lstrip("/"),
    user=_parsed.username,
    password=_parsed.password,
)
print(f"Connecting to {DB['host']}:{DB['port']}/{DB['dbname']} as {DB['user']}")

# COMMAND ----------

# DBTITLE 1,Connection helper (search_path -> weather schema)
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


def get_conn():
    """psycopg2 connection with search_path set to the `weather` schema, so
    unqualified table names resolve there first (matches lakebase.py)."""
    conn = psycopg2.connect(
        host=DB["host"], port=DB["port"], dbname=DB["dbname"],
        user=DB["user"], password=DB["password"],
        sslmode="require", connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute("SET search_path TO weather, public")
    return conn


# Smoke test: confirm we can reach the source table and see how much work there is.
with get_conn() as _c, _c.cursor(cursor_factory=RealDictCursor) as _cur:
    _cur.execute(f"SELECT COUNT(*) AS n FROM {DOCUMENTS_TABLE}")
    print(f"✅ Connected. {DOCUMENTS_TABLE} has {_cur.fetchone()['n']} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunking
# MAGIC Sliding character window. Most forecast periods are short (one chunk);
# MAGIC combined alert `description` + `instruction` text is where chunking earns
# MAGIC its keep. Overlap preserves context across chunk boundaries so a sentence
# MAGIC split down the middle still embeds coherently on both sides.

# COMMAND ----------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into overlapping windows of `size` chars, stepping by
    (size - overlap). Returns [text] unchanged when it already fits in one
    chunk. Whitespace-only tails are dropped."""
    text = (text or "").strip()
    if len(text) <= size:
        return [text] if text else []

    step = max(1, size - overlap)  # guard against overlap >= size
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


# Quick self-check of the chunker (runs on the driver, no DB needed):
_demo = "A" * 850
print(f"850 chars @ size={CHUNK_SIZE}/overlap={CHUNK_OVERLAP} -> {len(chunk_text(_demo))} chunks")
print(f"short text -> {chunk_text('Sunny, high near 78.')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the embedding model
# MAGIC Loaded once on the driver. all-MiniLM-L6-v2 is small and CPU-friendly, so
# MAGIC we don't need Spark to distribute embedding for these volumes.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Verify the model's real output dim matches the pgvector column width. Catching
# a mismatch HERE (cheap) beats a cryptic insert error against vector(384).
_probe_dim = len(model.encode("dimension probe"))
assert _probe_dim == EMBEDDING_DIM, (
    f"Model emits {_probe_dim}-dim vectors but config/columns expect {EMBEDDING_DIM}. "
    "Update EMBEDDING_DIM and the vector(N) columns in sql/ to match."
)
print(f"✅ Model loaded; confirmed {_probe_dim}-dim output.")


def to_vector_literal(vec) -> str:
    """Render an embedding as a pgvector text literal: '[0.1,0.2,...]'.

    We build the literal ourselves and cast with `%s::vector` in SQL rather
    than relying on a registered pgvector adapter — it works everywhere and
    makes the cast explicit and reviewable.
    """
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read documents to embed
# MAGIC We embed everything each run for simplicity; `ON CONFLICT` upserts make
# MAGIC that idempotent. To only embed NEW documents, add a LEFT JOIN anti-filter
# MAGIC against `weather_embeddings` here (noted in the README as an improvement).

# COMMAND ----------

with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(
        f"""
        SELECT id, location, headline, source_type, narrative_text
        FROM {DOCUMENTS_TABLE}
        WHERE narrative_text IS NOT NULL AND length(trim(narrative_text)) > 0
        ORDER BY synced_at
        """
    )
    documents = cur.fetchall()

print(f"Loaded {len(documents)} documents to embed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embed + write (document-level and chunk-level)
# MAGIC One pass builds both tables' rows in memory, then two `execute_values`
# MAGIC batched upserts push them to Postgres. `execute_values` collapses N rows
# MAGIC into ONE round-trip — dramatically faster than per-row `execute`.

# COMMAND ----------

doc_rows = []    # -> weather_embeddings
chunk_rows = []  # -> weather_chunk_embeddings

for doc in documents:
    doc_id = doc["id"]
    narrative = doc["narrative_text"]

    # (a) Document-level: embed headline + narrative as a single unit.
    doc_input = f"{doc['headline'] or ''}\n\n{narrative}".strip()
    doc_vec = model.encode(doc_input)
    doc_rows.append((
        doc_id, doc["location"], doc["headline"], doc["source_type"],
        to_vector_literal(doc_vec), EMBEDDING_MODEL_NAME,
    ))

    # (b) Chunk-level: split narrative, embed each window.
    chunks = chunk_text(narrative)
    if chunks:
        chunk_vecs = model.encode(chunks)  # batch-encode all chunks at once
        for idx, (chunk, vec) in enumerate(zip(chunks, chunk_vecs)):
            chunk_rows.append((
                f"{doc_id}:{idx}", doc_id, idx, chunk,
                to_vector_literal(vec), EMBEDDING_MODEL_NAME,
            ))

print(f"Prepared {len(doc_rows)} doc vectors and {len(chunk_rows)} chunk vectors.")

# COMMAND ----------

# DBTITLE 1,Batched upsert into both vector tables
with get_conn() as conn:
    with conn.cursor() as cur:
        # weather_embeddings: PK is document_id, so conflicts overwrite the vector.
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
            # NOTE the %s::vector cast on the 5th column — this is what turns our
            # text literal into a real pgvector value on insert.
            template="(%s, %s, %s, %s, %s::vector, %s)",
            page_size=BATCH_SIZE,
        )

        # weather_chunk_embeddings: PK is "<doc>:<idx>"; also UNIQUE(doc, idx).
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
                page_size=BATCH_SIZE,
            )
    conn.commit()

print(f"✅ Upserted {len(doc_rows)} doc vectors and {len(chunk_rows)} chunk vectors.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate
# MAGIC Confirm every document got a doc-level vector, and inspect chunk fan-out.
# MAGIC Expected: `doc_embeddings == documents`, `chunk_embeddings >= documents`.

# COMMAND ----------

with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(
        f"""
        SELECT 'documents'        AS layer, COUNT(*) AS n FROM {DOCUMENTS_TABLE}
        UNION ALL
        SELECT 'doc_embeddings',            COUNT(*)      FROM {EMBEDDINGS_TABLE}
        UNION ALL
        SELECT 'chunk_embeddings',          COUNT(*)      FROM {CHUNK_EMBEDDINGS_TABLE}
        """
    )
    for row in cur.fetchall():
        print(f"  {row['layer']:18s} {row['n']}")

# COMMAND ----------

# DBTITLE 1,Retrieval smoke test — does cosine search return sensible hits?
# End-to-end proof the vectors are queryable: embed a natural-language query and
# run the SAME cosine search the Flask /weather/search endpoint uses.
_q = "flash flood risk near rivers this weekend"
_q_vec = to_vector_literal(model.encode(_q))

with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(
        f"""
        SELECT d.location, d.headline, d.source_type,
               left(e.chunk_text, 100) AS preview,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {CHUNK_EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT 5
        """,
        (_q_vec, _q_vec),
    )
    print(f"Top matches for: {_q!r}\n")
    for r in cur.fetchall():
        print(f"  [{r['similarity']:.3f}] {r['location']} — {r['headline']} ({r['source_type']})")
        print(f"          {r['preview']}...")