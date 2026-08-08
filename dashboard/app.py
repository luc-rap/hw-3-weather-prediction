"""
Databricks Weather App:
- Serves a Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls weather data from the National Weather Service API via weather_client.py

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os

from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer

import lakebase
from weather_client import WeatherClient, geocode_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
EMBEDDINGS_TABLE_NAME = os.environ.get("EMBEDDINGS_TABLE_NAME", "weather_embeddings")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Match ingestion model

# Load embedding model once at module level (not per-request)
logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info("Embedding model loaded successfully")


def ensure_weather_table():
    """
    Create the weather documents table in Lakebase if it doesn't exist yet.
    This is the RAW document store for weather alerts, forecasts, and discussions.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            source_type TEXT NOT NULL,
            headline TEXT,
            event TEXT,
            narrative_text TEXT NOT NULL,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Render the weather harvester UI."""
    return render_template("index.html")


@app.route("/weather", methods=["GET"])
def list_weather():
    """Read weather documents already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    source_type = request.args.get("source_type")  # Optional filter: alert, forecast, discussion
    location = request.args.get("location")  # Optional filter by location
    
    query = f"SELECT * FROM {WEATHER_TABLE_NAME} WHERE 1=1"
    params = []
    
    if source_type:
        query += " AND source_type = %s"
        params.append(source_type)
    
    if location:
        query += " AND location = %s"
        params.append(location)
    
    query += " ORDER BY synced_at DESC LIMIT %s"
    params.append(limit)
    
    rows = lakebase.run_query(query, tuple(params))
    return jsonify(rows)


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Fetch weather data (state alerts, forecasts, hourly forecasts, discussions) for locations
    from the National Weather Service API and upsert into the weather_documents table.
    
    State-wide alerts are fetched once per unique state (e.g., all alerts for FL, TX).

    Body (optional JSON): 
    {
        "locations": ["Chicago, IL", "Austin, TX"], 
        "limit": 50,  # Max forecast periods/alerts per location (default: 50)
        "include_hourly": true,  # Include hourly forecasts (default: true)
        "include_discussion": false  # Include forecast discussions (default: false)
    }
    Each location is geocoded to lat/lon, then weather data is fetched.
    """
    # ensure_weather_table()  # Commented out - create tables manually via SQL scripts
    client = WeatherClient()

    body = request.json if request.is_json else {}
    locations = body.get("locations", [])
    limit = int(body.get("limit", 50))
    include_hourly = body.get("include_hourly", True)  # Default: True (homework requirement)
    include_discussion = body.get("include_discussion", False)

    if not locations:
        return jsonify({"error": "No locations provided"}), 400

    total = 0
    geocoded_locations = []
    
    # Track unique states to fetch state-wide alerts once per state
    states_processed = set()

    for location in locations:
        if not isinstance(location, str) or not location.strip():
            continue

        # Try to geocode the location
        coords = geocode_location(location)
        if not coords:
            logger.warning(f"Failed to geocode location: {location}")
            continue

        lat, lon = coords
        geocoded_locations.append({"location": location, "lat": lat, "lon": lon})
        
        # Fetch state-wide alerts (once per state)
        state_code = _extract_state_code(location)
        if state_code and state_code not in states_processed:
            states_processed.add(state_code)
            try:
                state_alerts = client.get_state_alerts(state_code, location_name=location, limit=limit)
                total += _upsert_weather_batch(state_alerts)
                logger.info(f"Fetched {len(state_alerts)} state-wide alerts for {state_code}")
            except Exception as e:
                logger.exception(f"Error fetching state alerts for {state_code}: {e}")

        # Fetch location-specific weather data
        try:
            documents = client.get_weather_for_location(
                lat, lon,
                location_name=location,  # Pass the original city/state name
                include_alerts=True,
                include_forecast=True,
                include_hourly=include_hourly,
                include_discussion=include_discussion,
                limit=limit  # Limit number of forecast periods and alerts per location
            )
            total += _upsert_weather_batch(documents)
        except Exception as e:
            logger.exception(f"Error fetching weather for {location} ({lat},{lon}): {e}")
            continue

    return jsonify({
        "synced": total,
        "locations": geocoded_locations,
        "requested": len(locations)
    })


def _extract_state_code(location: str) -> str | None:
    """Extract state code from location string like 'Miami, FL' -> 'FL'."""
    parts = location.split(",")
    if len(parts) >= 2:
        state = parts[-1].strip().upper()
        # Simple validation: 2-letter state codes
        if len(state) == 2 and state.isalpha():
            return state
    return None


def _upsert_weather_batch(documents: list[dict]) -> int:
    """Upsert weather documents into the weather_documents table.

    Each document should have: id, location, latitude, longitude, source_type, 
    headline, event, narrative_text, issued_at, effective_at, payload, synced_at.
    """
    if not documents:
        return 0

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, latitude, longitude, source_type, headline, event,
                        narrative_text, issued_at, effective_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            latitude = EXCLUDED.latitude,
                            longitude = EXCLUDED.longitude,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            event = EXCLUDED.event,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        doc.get("id"),
                        doc.get("location"),
                        doc.get("latitude"),
                        doc.get("longitude"),
                        doc.get("source_type"),
                        doc.get("headline"),
                        doc.get("event"),
                        doc.get("narrative_text"),
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        doc.get("payload"),
                    ),
                )
                count += 1
            conn.commit()
    return count


@app.route("/weather/search", methods=["POST"])
def search_weather():
    """Semantic search over weather document embeddings using pgvector.
    
    Body (JSON):
    {
        "query": "risk of flooding near rivers",
        "top_k": 5  # Optional, default 5, clamped to 1-20
    }
    
    Returns top_k most similar weather document chunks with:
    - location, headline, chunk_text, similarity score
    - Sorted by similarity (descending)
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    body = request.json
    query_text = body.get("query", "").strip()
    top_k = body.get("top_k", 5)
    
    # Validate query
    if not query_text:
        return jsonify({"error": "Missing or empty 'query' field"}), 400
    
    # Clamp top_k to reasonable bounds (1-20)
    try:
        top_k = int(top_k)
        top_k = max(1, min(20, top_k))  # Clamp between 1 and 20
    except (ValueError, TypeError):
        return jsonify({"error": "'top_k' must be an integer"}), 400
    
    # Embed the query using the same model as ingestion
    try:
        query_embedding = embedding_model.encode(query_text).tolist()
    except NameError as e:
        logger.exception(f"Embedding model not loaded: {e}")
        return jsonify({"error": "Embedding model not loaded. Restart the app after installing sentence-transformers."}), 500
    except Exception as e:
        logger.exception(f"Error embedding query: {e}")
        return jsonify({"error": f"Failed to embed query: {str(e)}"}), 500
    
    # Format embedding as pgvector literal (square brackets, not curly braces)
    embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'
    
    # Run cosine similarity search using pgvector's <=> operator
    try:
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT 
                        d.id, 
                        d.location, 
                        d.headline, 
                        d.narrative_text,
                        e.chunk_text,
                        1 - (e.embedding <=> %s::vector) AS similarity
                    FROM {EMBEDDINGS_TABLE_NAME} e
                    JOIN {WEATHER_TABLE_NAME} d ON d.id = e.document_id
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, embedding_str, top_k)
                )
                
                rows = cur.fetchall()
                
                # Handle empty results (no embeddings in table)
                if not rows:
                    return jsonify({
                        "results": [],
                        "message": "No embeddings found. Run the embedding notebook to populate weather_embeddings table."
                    })
                
                # Format results (RealDictCursor returns dicts, not tuples)
                results = [
                    {
                        "document_id": row['id'],
                        "location": row['location'],
                        "headline": row['headline'],
                        "narrative_text": row['narrative_text'],
                        "chunk_text": row['chunk_text'],
                        "similarity": float(row['similarity'])
                    }
                    for row in rows
                ]
                
                return jsonify({
                    "query": query_text,
                    "top_k": top_k,
                    "results": results,
                    "count": len(results)
                })
    
    except Exception as e:
        logger.exception(f"Error during semantic search: {e}")
        error_msg = str(e)
        
        # Provide helpful error messages for common issues
        if "relation" in error_msg and "does not exist" in error_msg:
            return jsonify({"error": "weather_embeddings table not found. Create it by running the embedding notebook."}), 500
        elif "column" in error_msg and "does not exist" in error_msg:
            return jsonify({"error": f"Schema mismatch: {error_msg}"}), 500
        else:
            return jsonify({"error": f"Search failed: {error_msg if error_msg else 'Unknown error'}"}), 500


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")