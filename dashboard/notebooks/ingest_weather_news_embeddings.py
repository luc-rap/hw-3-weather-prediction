# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Generate Chunked Weather Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook generates semantic embeddings for weather data stored in Lakebase Postgres,
# MAGIC using a **chunking strategy** for fine-grained search.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads weather documents from the `weather_documents` table in Lakebase
# MAGIC    (populated by the Flask app's `/weather/sync` endpoint)
# MAGIC 2. Splits each document's `narrative_text` into overlapping chunks
# MAGIC    (default: 800 chars with 100 char overlap)
# MAGIC 3. Computes sentence embeddings for each chunk using a sentence-transformers model,
# MAGIC    distributed across Spark workers
# MAGIC 4. Writes chunk embeddings into a `weather_embeddings` table using the `pgvector`
# MAGIC    Postgres extension for semantic similarity search
# MAGIC
# MAGIC **Why chunking?**
# MAGIC - Enables fine-grained semantic search on **specific passages**, not just whole documents
# MAGIC - Users can retrieve the exact weather info that matches their query
# MAGIC - Overlapping chunks prevent context loss at boundaries
# MAGIC
# MAGIC The workflow enables semantic search over:
# MAGIC - Weather alerts with rich narrative descriptions
# MAGIC - Multi-day forecast narratives
# MAGIC - Hourly forecast details
# MAGIC - State-wide alert summaries
# MAGIC - Forecast discussions (Area Forecast Discussion / AFD)
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook.

# COMMAND ----------

# MAGIC %pip install -q sentence-transformers psycopg2-binary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -q --force-reinstall psycopg2-binary
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("weather_table_name", "weather_documents", "Source table (weather data)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "Text chunk size (characters)")
dbutils.widgets.text("chunk_overlap", "100", "Text chunk overlap (characters)")

WEATHER_TABLE_NAME = dbutils.widgets.get("weather_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces both
# MAGIC Spark's JDBC reader AND the raw JDBC connection helper below need
# MAGIC (url/user/password).

# COMMAND ----------

import base64
import re
from urllib.parse import urlparse, quote_plus

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract project name and branch name from hostname
# Format: ep-{branch-name}-{random}.{project-name}.{region}.cloud.databricks.com
hostname_parts = parsed.hostname.split('.')
if len(hostname_parts) >= 2:
    # Extract project name (second part)
    project_name = hostname_parts[1]
    # Extract branch name from first part (ep-{branch-name}-{random})
    branch_match = re.match(r'ep-([^-]+)', hostname_parts[0])
    branch_name = branch_match.group(1) if branch_match else 'production'
else:
    raise ValueError(f"Unexpected Lakebase hostname format: {parsed.hostname}")

# Build JDBC URL for reading only (writes will use Lakebase SDK)
jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{parsed.port or 5432}{parsed.path}"
print(f"Connecting to: {parsed.hostname}:{parsed.port or 5432}{parsed.path}")
print(f"Project: {project_name}, Branch: {branch_name}")

# Pass credentials and SSL settings in properties for JDBC reads
jdbc_properties = {
    "user": parsed.username,
    "password": parsed.password,
    "driver": "org.postgresql.Driver",
    "sslmode": "require",
}

db_host = parsed.hostname
db_name = parsed.path.lstrip('/')
print(f"Database: {db_name}")

# COMMAND ----------

# DBTITLE 1,Test JDBC Connection
# Test JDBC connection with embedded credentials
try:
    test_df = spark.read.jdbc(
        url=jdbc_url,
        table=WEATHER_TABLE_NAME,
        properties=jdbc_properties
    )
    count = test_df.count()
    print(f"✅ Connection successful! Found {count} rows in {WEATHER_TABLE_NAME}")
    test_df.show(5)
except Exception as e:
    print(f"❌ Connection failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, you must manually create the `weather_embeddings` table
# MAGIC in your Lakebase Postgres database.
# MAGIC
# MAGIC The weather_documents table should already exist (created by your app's SQL setup).
# MAGIC
# MAGIC **Run the SQL file:** `sql/02_setup_weather_embeddings_table.sql`
# MAGIC
# MAGIC This creates a chunked embeddings table:
# MAGIC - Each weather document can have multiple chunks
# MAGIC - Each chunk gets its own embedding vector
# MAGIC - Enables fine-grained semantic search on passages, not just whole documents
# MAGIC - Schema: `id`, `document_id` (FK), `chunk_index`, `chunk_text`, `embedding`, `model_name`, `created_at`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Weather Documents
# MAGIC
# MAGIC Weather data is already populated in the `weather_documents` table by the
# MAGIC Flask app's `/weather/sync` endpoint. This notebook reads that data and
# MAGIC computes embeddings on the narrative_text field.
# MAGIC
# MAGIC The weather documents include:
# MAGIC - **Alerts** (point-based and state-wide) with rich descriptions
# MAGIC - **Multi-day forecasts** with detailed narratives
# MAGIC - **Hourly forecasts** with conditions and temperatures
# MAGIC - **Forecast discussions** (AFD) from NWS offices

# COMMAND ----------

# Read weather documents from Lakebase
from pyspark.sql.functions import col

weather_df = spark.read.jdbc(
    url=jdbc_url,
    table=WEATHER_TABLE_NAME,
    properties=jdbc_properties
)

total_count = weather_df.count()
print(f"\u2705 Loaded {total_count} weather documents from {WEATHER_TABLE_NAME}")

# Show breakdown by source type
print("\nBreakdown by source_type:")
weather_df.groupBy("source_type").count().orderBy(col("count").desc()).show()

# Show sample records
print("\nSample weather documents:")
weather_df.select("id", "location", "source_type", "event", "headline").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Weather Documents for Embedding
# MAGIC
# MAGIC Filters weather documents to only those with meaningful narrative_text,
# MAGIC which will be the input for our embedding model. We'll embed the full
# MAGIC narrative since it contains the rich weather information users want to
# MAGIC search semantically.

# COMMAND ----------

# Filter to documents with meaningful narrative text
source_df = (
    weather_df
    .select(
        "id",
        "location",
        "source_type",
        "event",
        "headline",
        "narrative_text"
    )
    .filter("narrative_text IS NOT NULL AND trim(narrative_text) != ''")
)

source_count = source_df.count()
print(f"\u2705 Found {source_count} weather documents with narrative text")

if source_count == 0:
    print("\u26a0\ufe0f  No documents with narrative_text found!")
    print("   Make sure your weather app has synced data with narrative fields populated.")
    dbutils.notebook.exit("No documents to process")
else:
    print("\nSample documents:")
    display(source_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Chunk Documents into Overlapping Segments
# MAGIC %md
# MAGIC ## Chunk documents into overlapping text segments
# MAGIC
# MAGIC Concatenates each document's context fields (**location**, **source_type**, **event**)
# MAGIC with its **narrative_text**, then splits into overlapping chunks of `CHUNK_SIZE`
# MAGIC characters with `CHUNK_OVERLAP` shared between consecutive chunks.
# MAGIC
# MAGIC **Why include context fields?**
# MAGIC - Embeddings capture the location, event type, and source alongside the narrative
# MAGIC - Enables better semantic matching: "hurricane in Miami" finds Miami hurricane chunks
# MAGIC - Each chunk is self-contained with full context
# MAGIC
# MAGIC **Format:** `Location: {location} | Type: {source_type} | Event: {event} | {narrative_text}`
# MAGIC
# MAGIC This prevents context loss at chunk boundaries and enables fine-grained semantic
# MAGIC search — users can retrieve specific passages that match their query, not just
# MAGIC whole documents.

# COMMAND ----------

from typing import Iterator
import pandas as pd
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

chunks_schema = StructType([
    StructField("document_id", StringType(), False),
    StructField("chunk_index", IntegerType(), False),
    StructField("chunk_text", StringType(), False),
])

def chunk_text_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Concatenate location + source_type + event + narrative_text, then split into overlapping chunks."""
    for batch in iterator:
        out_doc_ids, out_chunk_indexes, out_chunk_texts = [], [], []
        
        for doc_id, location, source_type, event, narrative in zip(
            batch["id"], 
            batch["location"],
            batch["source_type"],
            batch["event"],
            batch["narrative_text"]
        ):
            # Build context-enriched text: location | source_type | event | narrative
            parts = []
            if location:
                parts.append(f"Location: {location}")
            if source_type:
                parts.append(f"Type: {source_type}")
            if event:
                parts.append(f"Event: {event}")
            if narrative:
                parts.append(narrative)
            
            full_text = " | ".join(parts)
            
            if not full_text or len(full_text.strip()) == 0:
                continue
            
            # Split into overlapping chunks
            chunk_idx = 0
            start = 0
            while start < len(full_text):
                end = min(start + CHUNK_SIZE, len(full_text))
                chunk = full_text[start:end].strip()
                
                if chunk:  # Only include non-empty chunks
                    out_doc_ids.append(doc_id)
                    out_chunk_indexes.append(chunk_idx)
                    out_chunk_texts.append(chunk)
                    chunk_idx += 1
                
                # Move forward by (chunk_size - overlap)
                if end >= len(full_text):
                    break
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
        
        yield pd.DataFrame({
            "document_id": out_doc_ids,
            "chunk_index": out_chunk_indexes,
            "chunk_text": out_chunk_texts,
        })

chunks_df = source_df.select(
    "id", "location", "source_type", "event", "narrative_text"
).mapInPandas(
    chunk_text_partitions, 
    schema=chunks_schema
)

chunk_count = chunks_df.count()
print(f"\u2705 Created {chunk_count} text chunks from {source_count} documents")
print(f"   Chunk format: Location | Type | Event | Narrative")
print(f"   Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
print(f"   Average chunks per document: {chunk_count / source_count:.1f}")

print("\nSample chunks:")
display(chunks_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings (distributed pandas UDF)
# MAGIC
# MAGIC Loads the sentence-transformers model once per executor process (not per
# MAGIC row) and applies it in batches via `mapInPandas`, which scales across
# MAGIC however many workers the cluster has.

# COMMAND ----------

from typing import Iterator

import pandas as pd
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

embeddings_schema = StructType(
    [
        StructField("document_id", StringType(), False),
        StructField("chunk_index", IntegerType(), False),
        StructField("chunk_text", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
    ]
)


def embed_chunk_partitions(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Runs once per Spark partition/task: load the model once, then embed
    every batch of text chunks handed to this partition."""
    import os
    from sentence_transformers import SentenceTransformer

    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    for batch in iterator:
        vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
        yield pd.DataFrame(
            {
                "document_id": batch["document_id"],
                "chunk_index": batch["chunk_index"],
                "chunk_text": batch["chunk_text"],
                "embedding": [v.tolist() for v in vectors],
            }
        )


embeddings_result_df = chunks_df.mapInPandas(embed_chunk_partitions, schema=embeddings_schema)

embeddings_computed = embeddings_result_df.count()
print(f"\u2705 Computed {embeddings_computed} chunk embeddings using {EMBEDDING_MODEL_NAME}")
print(f"   Vector dimension: {EMBEDDING_DIM}")
print(f"   Average embeddings per document: {embeddings_computed / source_count:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Lakebase
# MAGIC
# MAGIC We'll write the computed embeddings to the `weather_embeddings` table
# MAGIC using psycopg2. The `pgvector` extension must already be enabled in your
# MAGIC Lakebase database, and the destination table must exist with the correct
# MAGIC vector dimension.

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/02_setup_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/02_setup_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

pip install pg8000

# COMMAND ----------

# DBTITLE 1,Write Embeddings via Spark JDBC (no collect)
import pg8000.dbapi as pg8000
from pyspark.sql.functions import col, concat_ws, current_timestamp, lit

# Create composite id: document_id + '_' + chunk_index
embeddings_with_id = embeddings_result_df.withColumn(
    "id", concat_ws("_", col("document_id"), col("chunk_index").cast("string"))
)

# Add model_name and created_at columns
embeddings_with_meta = (
    embeddings_with_id
    .withColumn("model_name", lit(EMBEDDING_MODEL_NAME))
    .withColumn("created_at", current_timestamp())
)

# Collect embeddings to driver for batch insert
embeddings_rows = embeddings_with_meta.collect()

if len(embeddings_rows) > 0:
    print(f"Inserting {len(embeddings_rows)} chunk embeddings into {EMBEDDINGS_TABLE_NAME}...")

    conn = pg8000.connect(
        host=db_host,
        port=parsed.port or 5432,
        database=db_name,
        user=parsed.username,
        password=parsed.password,
        ssl_context=True,
    )

    try:
        cursor = conn.cursor()

        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
            ) VALUES (%s, %s, %s, %s, %s::double precision[], %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                created_at = EXCLUDED.created_at
        """

        insert_data = [
            (
                row.id,
                row.document_id,
                row.chunk_index,
                row.chunk_text,
                '{' + ','.join(str(float(x)) for x in row.embedding) + '}',
                row.model_name,
                row.created_at,
            )
            for row in embeddings_rows
        ]

        cursor.executemany(insert_sql, insert_data)
        conn.commit()

        print(f"✅ Successfully wrote {len(insert_data)} chunk embeddings to {EMBEDDINGS_TABLE_NAME}")
        print(f"   (Existing embeddings were updated)")
        print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
        print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector({EMBEDDING_DIM}) WHERE embedding IS NOT NULL;")

    finally:
        cursor.close()
        conn.close()
else:
    print("No embeddings to write.")