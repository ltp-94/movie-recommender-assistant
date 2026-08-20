import os

# ============================================================
# 1. DATABASE & CONNECTION SETTINGS
# ============================================================
INDEX_NAME = "movies"
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# ============================================================
# 2. INGESTION & DATA SETTINGS
# ============================================================
DATA_PATH = "../data/movies_with_embeddings.parquet"
BATCH_SIZE = 1500  # Number of documents per bulk request

# ============================================================
# 3. ELASTICSEARCH MAPPING (THE SCHEMA)
# ============================================================
def get_index_mapping(embedding_dim: int) -> dict:
    """
    Returns the structural blueprint for the Elasticsearch index.
    Organized into Text, Categorical, Numeric, and Vector fields.
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                # --- Core Content & Search ---
                "title":          {"type": "text"},
                "overview":       {"type": "text"},
                "search_context": {"type": "text"},
                "keywords":       {"type": "text"},

                # --- Categorical & Filtering ---
                "genres": {
                    "type": "keyword", 
                    "fields": {"text": {"type": "text"}}
                },
                "language":       {"type": "keyword"},
                "country":        {"type": "keyword"},
                "status":         {"type": "keyword"},

                # --- People & Production ---
                "director":       {"type": "text"},
                "writers":        {"type": "text"},
                "cast":           {"type": "text"},
                "production_companies": {"type": "text"},

                # --- Numeric Metadata & Ratings ---
                "runtime":        {"type": "integer"},
                "release_year":   {"type": "integer"},
                "popularity":     {"type": "float"},
                "vote_average":   {"type": "float"},  # TMDB Rating
                "imdb_rating":    {"type": "float"},
                "vote_count":     {"type": "integer"},

                # --- URLs & Assets (Not indexed for search) ---
                "poster_url":     {"type": "keyword", "index": False},
                "movie_link":     {"type": "keyword", "index": False},

                # --- Vector Search ---
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }