import os
import time
import torch
import math
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Modular imports
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
from rag.llm_query_rewriting import rewrite_query
from rag.llm_generation import generate_recommendation

load_dotenv()

# ============================================================
# 1. SETUP (Load models ONCE)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading search models...")
# Using st.cache_resource pattern logic or simple global variables
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# es = Elasticsearch(ELASTICSEARCH_URL)
# # Change this line:
# es = Elasticsearch(ELASTICSEARCH_URL)

# To this:
es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=60,       # Increased to 60 seconds
    retry_on_timeout=True, 
    max_retries=3
)



def rerank_movies(query, movies):
    if not movies:
        return []

    pairs = []
    for m in movies:
        context = f"Title: {m.get('title')}. Genres: {m.get('genres')}. Plot: {m.get('overview')}. Keywords: {m.get('keywords')}. Cast: {m.get('cast')}. Director: {m.get('director')}"
        pairs.append([query, context])
    
    scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
    
    for movie, score in zip(movies, scores):
        movie["cross_score"] = float(score)
        # Apply your popularity boost here if desired
        pop = movie.get("popularity", 0) or 0
        movie["final_score"] = movie["cross_score"] + (math.log(pop + 1) * 0.85)

    movies.sort(key=lambda x: x["final_score"], reverse=True)
    return movies