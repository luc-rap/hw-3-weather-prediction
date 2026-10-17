# Weather Data Integration with Semantic Search

## Data Source Selection

### National Weather Service (NWS) API

**Choice:** [National Weather Service API](https://www.weather.gov/documentation/services-web-api)

* **Free and open:** No API key required, no rate limits for reasonable use
* **Official and reliable:** Authoritative source from NOAA (National Oceanic and Atmospheric Administration)
* **Rich narrative content:** Provides detailed forecast discussions, area forecast discussions (AFD), and narrative weather descriptions ideal for semantic search

## Schema Design

### 1. `weather_documents` Table

Stores raw weather data synchronized from the NWS API.

```sql
CREATE TABLE "weather_documents" (
	"id" text PRIMARY KEY,
	"location" text NOT NULL,
	"latitude" double precision,
	"longitude" double precision,
	"source_type" text NOT NULL,
	"headline" text,
	"event" text,
	"narrative_text" text NOT NULL,
	"issued_at" timestamp with time zone,
	"effective_at" timestamp with time zone,
	"payload" jsonb NOT NULL,
	"synced_at" timestamp with time zone DEFAULT now() NOT NULL
);
CREATE UNIQUE INDEX "weather_documents_pkey" ON "weather_documents" ("id");
);
```

### 2. `weather_embeddings` Table

Stores chunked, embedded weather narratives for semantic search using pgvector.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE "weather_embeddings" (
	"id" text PRIMARY KEY,
	"document_id" text NOT NULL,
	"chunk_index" integer NOT NULL,
	"chunk_text" text NOT NULL,
	"embedding" vector(384) NOT NULL,
	"model_name" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT "weather_embeddings_document_id_chunk_index_key" UNIQUE("document_id","chunk_index")
);
CREATE UNIQUE INDEX "weather_embeddings_document_id_chunk_index_key" ON "weather_embeddings" ("document_id","chunk_index");
CREATE UNIQUE INDEX "weather_embeddings_pkey" ON "weather_embeddings" ("id");

CREATE INDEX IF NOT EXISTS weather_embeddings_vector_idx 
    ON weather_embeddings 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

**Index Strategy:**
* `ivfflat` index with cosine distance for approximate nearest neighbor search
* `lists = 100`: Balances query speed vs. accuracy for datasets of ~1K-10K embeddings
* Cosine distance (`<=>` operator) preferred over L2 for semantic similarity

## Chunking Strategy

**Parameters:**
* **Chunk size:** 800 characters
* **Overlap:** 100 characters


* **800 characters:** Balances context preservation with embedding model constraints
  - Captures 2-4 complete sentences (weather discussions are verbose)
  - Stays well below sentence-transformers' 512-token limit (~2000 characters)
  - Small enough to ensure focused semantic meaning per chunk
  - Large enough to maintain coherent weather context (e.g., "...cold front moving through the region tonight, bringing...")

* **100-character overlap:** Prevents loss of semantic context at chunk boundaries
  - Ensures multi-sentence concepts aren't artificially split
  - Allows queries matching boundary content to find relevant chunks
  - Small enough to avoid redundant storage


## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Specifications:**
* **Dimensions:** 384
* **Max sequence length:** 256 word pieces (~512 tokens, ~2000 characters)
* **Model size:** ~80MB
* **Framework:** Sentence-Transformers

**Why This Model:**

1. **Balance of quality and speed:**
   - High semantic understanding for general English text
   - Fast inference (~0.1s per batch on CPU)
   - Suitable for real-time query embedding

2. **Proven for semantic search:**
   - Pre-trained on 1B+ sentence pairs
   - Optimized for cosine similarity tasks
   - Widely used in production RAG/search systems

3. **Resource-efficient:**
   - 384 dimensions vs. 768+ in larger models
   - Faster pgvector index builds and queries
   - Lower storage overhead

4. **Open source and permissive:** Apache 2.0 license, no API costs

## End-to-End Pipeline

### Step 1: Sync Weather Data

Fetch weather data from NWS API and store in `weather_documents`:

**What it does:**
1. Fetches active weather alerts for the state
2. For each alert location:
   - Gets detailed forecast (7-day)
   - Gets hourly forecast (48-hour)
   - Gets area forecast discussion (AFD) - the narrative text
3. Upserts into `weather_documents` (by location)


### Step 2: Generate Embeddings

Chunk narrative texts and generate vector embeddings:

1. Open `ingest_weather_news_embeddings.ipynb` 
2. Attach to any cluster with Python support
3. Run all cells


**What it does:**
1. Reads all records from `weather_documents`
2. Chunks each `narrative_text` into 800-char segments with 100-char overlap
3. Loads `sentence-transformers/all-MiniLM-L6-v2` model
4. Generates 384-dim embeddings for each chunk
5. Batch inserts into `weather_embeddings` with pgvector `vector(384)` type

**Output:** Populated `weather_embeddings` table ready for semantic search

### Step 3: Semantic Search

Query weather data by natural language meaning:

**Request:**
* `query` (required): Natural language search query
* `limit` (optional, default=10, max=20): Number of results to return

**What it does:**
1. Embeds the query using the same sentence-transformers model
2. Searches `weather_embeddings` using pgvector's `<=>` (cosine distance) operator
3. Returns top-K most semantically similar chunks with similarity scores

## Known Limitations and Future Improvements

### Current Limitations

1. **No incremental updates:**
   - Each sync overwrites previous data for a location
   - No historical tracking of forecast changes
   - **Improvement:** Add versioning with `synced_at` as part of composite key, keep history

2. **No real-time updates:**
   - Manual sync required; no webhook or polling
   - **Improvement:** Schedule sync job every 1-3 hours, add staleness indicators in UI

### Potential Enhancements

1. **Time-aware search:** Filter by forecast time range ("weekend outlook", "tonight's weather")
2. **Alert notifications:** Trigger alerts when new high-severity events are synced
3. **Comparison view:** "Compare forecasts for Tampa vs. Orlando"
4. **RAG chatbot:** Feed search results into an LLM for conversational weather Q&A

## Architecture Diagram

```
┌─────────────────┐
│  NWS API        │
│  weather.gov    │
└────────┬────────┘
         │ HTTP GET
         ▼
┌─────────────────────────┐
│  weather_client.py      │  ← Fetch alerts, forecasts, discussions
│  (Flask app.py)         │
└────────┬────────────────┘
         │ INSERT/UPDATE
         ▼
┌─────────────────────────┐
│  Lakebase Postgres      │
│  weather_documents      │  ← Raw weather data
│  (location, headline,   │
│   narrative_text, ...)  │
└────────┬────────────────┘
         │ ETL (notebook)
         ▼
┌─────────────────────────┐
│  ingest_weather_news    │  ← Chunk + Embed
│  _embeddings.ipynb      │
└────────┬────────────────┘
         │ INSERT
         ▼
┌─────────────────────────┐
│  Lakebase Postgres      │
│  weather_embeddings     │  ← Chunked text + vectors
│  (chunk_text,           │
│   embedding vector(384))│
└────────┬────────────────┘
         │ SELECT ... ORDER BY embedding <=> $query
         ▼
┌─────────────────────────┐
│  POST /weather/search   │  ← Semantic search API
│  (Flask app.py)         │
└────────┬────────────────┘
         │ JSON response
         ▼
┌─────────────────────────┐
│  UI (index.html)        │  ← Display results
│  Semantic Search form   │
└─────────────────────────┘
```
