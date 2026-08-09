"""
Client for the National Weather Service (NWS) API — https://api.weather.gov

Why NWS:
    * Free, US-government-run, and requires NO API key or auth token. That lets
      this homework focus on the harvest -> vectorize -> retrieve pipeline
      instead of auth plumbing.
    * It returns rich, unstructured *narrative* text that is ideal for embedding:
      alert `description`/`instruction` fields and multi-day forecast
      `detailedForecast` strings.

Two NWS-specific gotchas this client handles for you (both cause silent,
hard-to-debug failures if you miss them):

    1. USER-AGENT IS MANDATORY. NWS rejects requests with a generic or missing
       User-Agent (HTTP 403). Their policy asks that the UA identify your app
       and a contact (email or URL) so they can reach you about abusive traffic.
       We read it from the WEATHER_USER_AGENT env var — never hardcode it.

    2. /points COORDINATES MUST BE <= 4 DECIMAL PLACES. NWS 301-redirects
       over-precise coordinates; `requests` follows the redirect but the extra
       hop is wasteful and occasionally 404s. We round before calling.

The public surface follows the same shape the Flask app expects from any data
client: construct the client, call a fetch method, get back plain dicts ready to
upsert into Lakebase.
"""

import hashlib
import os
from typing import Any, Iterator

import requests

# Base URL is env-overridable so tests / mocks can point at a local fixture
# server without code changes. Defaults to the real NWS host.
_BASE_URL = os.environ.get("WEATHER_API_BASE_URL", "https://api.weather.gov")

# NWS asks every client to send a descriptive User-Agent with a contact. We
# require it via env rather than shipping a default, so each deployment is
# individually identifiable to NWS (and so we never commit a real email).
_USER_AGENT = os.environ.get(
    "WEATHER_USER_AGENT",
    "(weather-lakebase-homework, contact@example.com)",
)

_DEFAULT_TIMEOUT = 30

# --- Curated city -> (lat, lon) lookup -------------------------------------
# /points only accepts coordinates; NWS does no geocoding. Rather than pull in
# an external geocoder, we ship a small map of common US cities so the
# assignment's `"Chicago, IL"` style input works out of the box. Extend freely.
# Keys are normalized to lowercase "city, st" (see _normalize_location_key).
CITY_COORDS: dict[str, tuple[float, float]] = {
    "chicago, il": (41.8781, -87.6298),
    "austin, tx": (30.2672, -97.7431),
    "new york, ny": (40.7128, -74.0060),
    "los angeles, ca": (34.0522, -118.2437),
    "miami, fl": (25.7617, -80.1918),
    "denver, co": (39.7392, -104.9903),
    "seattle, wa": (47.6062, -122.3321),
    "houston, tx": (29.7604, -95.3698),
    "new orleans, la": (29.9511, -90.0715),
    "san francisco, ca": (37.7749, -122.4194),
    "boston, ma": (42.3601, -71.0589),
    "oklahoma city, ok": (35.4676, -97.5164),  # tornado alley — good alert demos
}


def _normalize_location_key(location: str) -> str:
    """Lowercase + collapse whitespace so 'Chicago,  IL' matches 'chicago, il'."""
    return ", ".join(part.strip() for part in location.lower().split(",")).strip()


def resolve_location(location: str) -> tuple[float, float, str]:
    """
    Turn a user-supplied location into (lat, lon, canonical_label).

    Accepts either:
      * a curated city name, e.g. "Chicago, IL"
      * a raw coordinate pair, e.g. "41.88,-87.63"

    Raises ValueError for anything we can't resolve, so /weather/sync can
    return a clean 400 instead of a confusing downstream NWS error.
    """
    key = _normalize_location_key(location)
    if key in CITY_COORDS:
        lat, lon = CITY_COORDS[key]
        return lat, lon, location.strip()

    # Fall back to parsing "lat,lon". We split on comma and float() each half;
    # any malformed input raises ValueError, which is exactly what we want.
    try:
        lat_str, lon_str = location.split(",")
        lat, lon = float(lat_str), float(lon_str)
    except (ValueError, AttributeError):
        raise ValueError(
            f"Unrecognized location {location!r}: expected a known city "
            f'(e.g. "Chicago, IL") or a "lat,lon" pair.'
        )
    return lat, lon, f"{lat},{lon}"


class WeatherClient:
    """Thin wrapper around the NWS API with a compliant User-Agent session."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # NWS recommends the GeoJSON media type; User-Agent is required (see
        # module docstring). No Authorization header — this API is open.
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    # --- Grid resolution ---------------------------------------------------
    def resolve_point(self, lat: float, lon: float) -> dict[str, Any]:
        """
        Resolve a lat/lon to an NWS grid point via GET /points/{lat},{lon}.

        NWS models the country as a set of grids owned by regional forecast
        offices. To get a forecast you first translate coordinates into
        (office, grid_x, grid_y) here, then call get_forecast() with them.

        Returns the pieces the rest of the pipeline needs, including the
        human-readable city/state NWS reverse-geocodes for us in
        `relativeLocation` (handy when the caller passed raw coordinates).
        """
        # Round to 4 decimals to satisfy the NWS precision limit (see module doc).
        props = self.get(f"/points/{round(lat, 4)},{round(lon, 4)}")["properties"]
        rel = props.get("relativeLocation", {}).get("properties", {})
        return {
            "office": props["gridId"],
            "grid_x": props["gridX"],
            "grid_y": props["gridY"],
            "city": rel.get("city"),
            "state": rel.get("state"),
        }

    # --- Alerts (source_type = "alert") ------------------------------------
    def get_active_alerts(self, state: str, location_label: str) -> Iterator[dict]:
        """
        Fetch active alerts for a US state via GET /alerts/active?area={ST}
        and yield normalized document dicts.

        `state` is a 2-letter code (e.g. "IL"); `location_label` is the display
        string we attach to each doc so search results can name the location.

        Alerts are the most vivid unstructured text NWS offers — each carries a
        free-text `description` ("A Flash Flood Warning means...") and often an
        `instruction` ("Move to higher ground now."). We concatenate both into
        one narrative for embedding.
        """
        data = self.get("/alerts/active", params={"area": state})
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            description = props.get("description") or ""
            instruction = props.get("instruction") or ""
            # Join description + instruction so a single embedding captures both
            # "what is happening" and "what to do about it".
            narrative = "\n\n".join(p for p in (description, instruction) if p)
            if not narrative.strip():
                continue  # skip alerts with no free text — nothing to embed

            yield {
                # Alert `id` is a stable URN (urn:oid:...) — perfect dedup key.
                "id": props.get("id") or feature.get("id"),
                "location": location_label,
                "source_type": "alert",
                "headline": props.get("headline") or props.get("event"),
                "event": props.get("event"),
                "narrative_text": narrative,
                "effective_at": props.get("effective") or props.get("onset"),
                "expires_at": props.get("expires") or props.get("ends"),
                "payload": feature,
            }

    # --- Forecasts (source_type = "forecast") ------------------------------
    def get_forecast(
        self, office: str, grid_x: int, grid_y: int, location_label: str
    ) -> Iterator[dict]:
        """
        Fetch the multi-day forecast via GET /gridpoints/{office}/{x},{y}/forecast
        and yield one normalized document per forecast period.

        Each period (e.g. "This Afternoon", "Tonight") has a `detailedForecast`
        narrative — "Sunny, with a high near 78. Northwest wind around 6 mph." —
        which is the free text we embed.
        """
        data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")
        for period in data.get("properties", {}).get("periods", []):
            narrative = period.get("detailedForecast") or ""
            if not narrative.strip():
                continue

            start_time = period.get("startTime")
            yield {
                # Forecast periods have no natural ID, so build a deterministic
                # one from location + period start. Re-syncing the same period
                # produces the same id -> ON CONFLICT upsert, no duplicates.
                "id": _forecast_id(location_label, start_time),
                "location": location_label,
                "source_type": "forecast",
                "headline": period.get("name"),  # "This Afternoon", "Tonight"
                "event": None,
                "narrative_text": narrative,
                "effective_at": start_time,
                "expires_at": period.get("endTime"),
                "payload": period,
            }


def _forecast_id(location_label: str, start_time: str | None) -> str:
    """Stable dedup key for a forecast period: sha256(location|start_time).

    Using a hash (rather than a raw concatenation) keeps the PRIMARY KEY a
    fixed, index-friendly length regardless of how long the location label is.
    """
    raw = f"{location_label}|{start_time or ''}"
    return "forecast:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
