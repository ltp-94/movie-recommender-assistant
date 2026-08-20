# import os
# import polars as pl
# from elasticsearch import Elasticsearch, helpers

# # Import settings and schema from your config
# from config import (
#     INDEX_NAME, 
#     ELASTICSEARCH_URL, 
#     DATA_PATH, 
#     BATCH_SIZE, 
#     get_index_mapping
# )

# # Import your pre-processing logic
# from data_utils import prepare_data

# # Initialize the Elasticsearch client
# es = Elasticsearch(
#     ELASTICSEARCH_URL,
#     request_timeout=120,
#     retry_on_timeout=True
# )

# def run_ingestion(parquet_path: str):
#     """
#     Main pipeline: Loads Parquet, cleans data, recreates index, 
#     and performs bulk ingestion into Elasticsearch.
#     """
#     # --- 1. LOAD DATA ---
#     if not os.path.exists(parquet_path):
#         print(f"❌ Error: Data file not found at {parquet_path}")
#         return

#     print(f"📂 Loading raw data from: {parquet_path}")
#     raw_df = pl.read_parquet(parquet_path)

#     # --- 2. PRE-PROCESS DATA ---
#     # This calls your function from data_utils.py to create URLs and Years
#     df = prepare_data(raw_df)
    
#     # --- 3. DYNAMIC METADATA ---
#     # Identify the correct embedding column
#     #emb_col = "embedding" if "embedding" in df.columns else "embeddings"
    
#     # Get embedding dimension from the first row
#     embedding_sample = df["embedding"][0]
#     dim = len(embedding_sample)
    
#     print(f"✅ Data prepared. Detected {len(df)} movies.")
#     print(f"🧬 Embedding Dimension: {dim}")

#     # --- 4. RECREATE ELASTICSEARCH INDEX ---
#     print(f"🗑️  Deleting old index (if exists): {INDEX_NAME}")
#     es.indices.delete(index=INDEX_NAME, ignore_unavailable=True)

#     print(f"🏗️  Creating new index with mapping: {INDEX_NAME}")
#     mapping = get_index_mapping(dim)
#     es.indices.create(index=INDEX_NAME, body=mapping)

#     # --- 5. PREPARE BULK ACTIONS ---
#     print(f"📝 Preparing bulk actions...")
#     actions = []
    
#     # Convert Polars DataFrame to a list of dictionaries for easier mapping
#     movies_list = df.to_dicts()

#     for row in movies_list:
#         # Create the Elasticsearch action document
#         action = {
#             "_index": INDEX_NAME,
#             "_source": {
#                 # Text Search Fields
#                 "title":          row.get("title"),
#                 "overview":       row.get("overview"),
#                 "search_context": row.get("input") or row.get("search_context"),
#                 "keywords":       row.get("keywords"),
                
#                 # Categorical / Filtering
#                 "genres":         row.get("genres"),
#                 "language":       row.get("original_language"),
#                 "country":        row.get("production_countries"),
                
#                 # People & Production
#                 "director":       row.get("director"),
#                 "writers":        row.get("writers"),
#                 "cast":           row.get("cast"),

#                 # Numeric Metadata
#                 "runtime":        row.get("runtime"),
#                 "release_year":   row.get("release_year"),
#                 "popularity":     row.get("popularity"),
#                 "vote_average":   row.get("vote_average"),
#                 "imdb_rating":    row.get("imdb_rating"),
#                 "vote_count":     row.get("vote_count"),

#                 # URLs (Constructed in data_utils.py)
#                 "poster_url":     row.get("poster_url"),
#                 "movie_link":     row.get("movie_link"),

#                 # Vector Field
#                 "embedding":      row.get("embedding")
#             }
#         }
#         actions.append(action)

#     # --- 6. EXECUTE BULK UPLOAD ---
#     print(f"🚀 Starting bulk upload (Batch size: {BATCH_SIZE})...")
    
#     try:
#         success, failed = helpers.bulk(
#             es, 
#             actions, 
#             chunk_size=BATCH_SIZE, 
#             stats_only=False
#         )
        
#         # Refresh the index to make data searchable immediately
#         es.indices.refresh(index=INDEX_NAME)
        
#         final_count = es.count(index=INDEX_NAME)["count"]
#         print(f"\n✨ Ingestion Successful!")
#         print(f"✅ Documents indexed: {final_count}")
        
#         if failed:
#             print(f"⚠️  Warning: {len(failed)} documents failed to index.")

#     except Exception as e:
#         print(f"❌ Critical Error during bulk upload: {e}")

# # ============================================================
# # ENTRY POINT
# # ============================================================
# if __name__ == "__main__":
#     # Ensure you are running this from the project root or the paths are correct
#     run_ingestion(DATA_PATH)


import os
import polars as pl
from elasticsearch import Elasticsearch, helpers

# Import settings from your config
from config import INDEX_NAME, ELASTICSEARCH_URL, DATA_PATH, BATCH_SIZE
# Import your pre-processing logic
from data_utils import prepare_data

# Initialize Elasticsearch
es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=120,
    retry_on_timeout=True
)

def run_ingestion(parquet_path: str):
    # 1. LOAD DATA
    if not os.path.exists(parquet_path):
        print(f"❌ Error: Data file not found at {parquet_path}")
        return

    print(f"📂 Loading data from: {parquet_path}")
    raw_df = pl.read_parquet(parquet_path)
    
    # 2. PRE-PROCESS (Constructs URLs, Years, etc.)
    df = prepare_data(raw_df)
    total_docs = len(df)
    
    print(f"✅ Data prepared. Total movies to index: {total_docs}")

    # 3. CHUNKED INGESTION LOOP
    print(f"🚀 Starting ingestion in chunks of {BATCH_SIZE}...")

    for start in range(0, total_docs, BATCH_SIZE):
        # Calculate end of the current chunk
        end = min(start + BATCH_SIZE, total_docs)
        
        # Take a slice of the Polars DataFrame for this batch
        # This is memory efficient because it doesn't duplicate the whole DF
        chunk = df.slice(start, BATCH_SIZE).to_dicts()
        
        actions = []
        for row in chunk:
            # Map the flat Parquet row to the Elasticsearch document structure
            actions.append({
                "_index": INDEX_NAME,
                "_source": {
                    "title":          row.get("title"),
                    "overview":       row.get("overview"),
                    "search_context": row.get("input"), 
                    "keywords":       row.get("keywords"),
                    "genres":         row.get("genres"),
                    "director":       row.get("director"),
                    "writers":        row.get("writers"),
                    "cast":           row.get("cast"),
                    "language":       row.get("original_language"),
                    "country":        row.get("production_countries"),
                    "status":         row.get("status"),
                    "runtime":        row.get("runtime"),
                    "release_year":   row.get("release_year"),
                    "popularity":     row.get("popularity"),
                    "vote_average":   row.get("vote_average"),
                    "imdb_rating":    row.get("imdb_rating"),
                    "vote_count":     row.get("vote_count"),
                    "poster_url":     row.get("poster_url"),
                    "movie_link":     row.get("movie_link"),
                    
                    # THE FIX: Ensure key is 'embedding' (not 'embeddding')
                    "embedding":      row.get("embedding") 
                }
            })

        # Bulk upload the chunk to Elasticsearch
        if actions:
            helpers.bulk(es, actions, stats_only=True)
        
        print(f"📦 Progress: {end}/{total_docs} movies indexed...")

    # 4. FINALIZE
    es.indices.refresh(index=INDEX_NAME)
    final_count = es.count(index=INDEX_NAME)["count"]
    print(f"\n✨ Ingestion Successful!")
    print(f"🏁 Total documents now in '{INDEX_NAME}': {final_count}")

if __name__ == "__main__":
    # Set PYTHONPATH=. in your terminal to ensure imports work
    run_ingestion(DATA_PATH)