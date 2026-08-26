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





def reciprocal_rank_fusion(bm25_hits, vector_hits, k=60, top_n=150):
    scores = {}
    documents = {}

    W_VECTOR = 0.7
    W_BM25 = 0.3

    # Process BM25
    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0) + (W_BM25 / (k + rank))
        documents[doc_id] = hit["content"]

    # Process Vector
    for rank, hit in enumerate(vector_hits, start=1):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0) + (W_VECTOR / (k + rank))
        documents[doc_id] = hit["content"]

    # Sort by RRF Score
    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for doc_id, score in ranked_ids[:top_n]:
        doc = documents[doc_id].copy()
        doc["rrf_score"] = score
        doc["_id"] = doc_id
        results.append(doc)
    
    return results