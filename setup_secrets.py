"""
One-time setup: create the Databricks secret scope that holds the Lakebase
connection URL, and grant workspace users read access.

The weather app needs NO third-party API key — NWS (api.weather.gov) is open,
authenticated only by a descriptive User-Agent (set via env, not a secret). So
the single secret this app requires is the Lakebase Postgres URL.

Store the URL base64-encoded (that's how lakebase.py decodes it):
    printf 'postgresql://role:pw@host:5432/db?sslmode=require' | base64

Usage (with the Databricks CLI configured):
    python setup_secrets.py
Never commit the resulting secret value anywhere.
"""
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

w = WorkspaceClient()

SCOPE = "database-day2"  # matches LAKEBASE_SECRET_SCOPE in app.yaml / .env

# create_scope raises if the scope already exists; ignore that so re-runs are safe.
try:
    w.secrets.create_scope(scope=SCOPE)
except Exception as exc:
    print(f"Scope {SCOPE!r} may already exist ({exc}); continuing.")

w.secrets.put_secret(
    scope=SCOPE,
    key="lakebase-url",
    string_value=getpass.getpass("Paste your base64-encoded Lakebase URL: "),
)

w.secrets.put_acl(
    scope=SCOPE,
    principal="users",
    permission=workspace.AclPermission.READ,
)

print(f"✅ Stored 'lakebase-url' in scope {SCOPE!r} with read access for users.")
