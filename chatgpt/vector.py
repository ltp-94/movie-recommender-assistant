import os
import time
import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Modular imports
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
from rag.llm_query_rewriting import rewrite_query

load_dotenv()

# ============================================================
# SETUP MODELS & ES
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Bi-Encoder (Ensure path is correct or use HuggingFace name)
# Using HuggingFace name for portability
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=60,
    retry_on_timeout=True
)

# ============================================================
# VECTOR SEARCH FUNCTION
# ============================================================

def vector_search(original_query, top_n=10):
    """
    Performs a pure vector (KNN) search.
    """
    # 1. Rewrite Query
    rewritten_query = rewrite_query(original_query)
    
    # 2. Encode to Vector
    # BGE models work best with this instruction prefix
    instruction = "Represent this sentence for searching relevant movie plots: "
    query_vector = bi_encoder.encode(
        instruction + rewritten_query, 
        normalize_embeddings=True
    ).tolist()

    # 3. Build KNN Search Body
    search_body = {
        "size": top_n,
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_n,
            "num_candidates": 300
        },
        "_source": [
            "title", "overview", "genres", "director", "writers", 
            "cast", "keywords", "popularity", "vote_average", 
            "poster_url", "movie_link", "release_year", "runtime"
        ]
    }

    # 4. Execute Search
    response = es.search(index=INDEX_NAME, body=search_body)
    
    hits = []
    for hit in response["hits"]["hits"]:
        doc = hit["_source"]
        doc["vector_score"] = hit["_score"] # This is the cosine similarity score
        hits.append(doc)
        
    return hits

# ============================================================
# RUN TEST
# ============================================================
if __name__ == "__main__":
    test_query = "A psychological thriller where a man has no short term memory"
    test_query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion"
    
    print(f"📡 Testing Vector Search for: '{test_query}'\n")
    start = time.time()
    results = vector_search(test_query, top_n=30)
    
    if not results:
        print("❌ No results found.")
    else:
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} (Score: {movie['vector_score']:.4f})")
            print(f"   Overview: {movie.get('overview', '')[:100]}...")
            print("-" * 50)
    
    print(f"Search completed in {time.time() - start:.2f}s")