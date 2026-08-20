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

load_dotenv()

# ============================================================
# 1. SETUP (Load models ONCE)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading search models...")
# Using st.cache_resource pattern logic or simple global variables
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

es = Elasticsearch(ELASTICSEARCH_URL)

# ============================================================
# 2. UPDATED SEARCH FUNCTIONS (Returning IDs)
# ============================================================

def get_bm25_results(query, top_n=100):
    search_body = {
        "size": top_n,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["genres.text", "overview^2", "cast^3", "director", "search_context^3"],
                "type": "best_fields"
            }
        }
    }
    response = es.search(index=INDEX_NAME, body=search_body)
    # We return a list of dicts that include BOTH the ID and the Source
    return [{"id": hit["_id"], "content": hit["_source"]} for hit in response["hits"]["hits"]]

def get_vector_results(query, top_n=100):
    # Rewriting happens inside the high-level flow, not here
    instruction = "Represent this sentence for searching relevant movie plots: "
    query_vector = bi_encoder.encode(instruction + query, normalize_embeddings=True).tolist()

    response = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_n,
            "num_candidates": 200
        }
    )
    return [{"id": hit["_id"], "content": hit["_source"]} for hit in response["hits"]["hits"]]

# ============================================================
# 3. RRF LOGIC
# ============================================================

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

# ============================================================
# 4. RERANKING LOGIC
# ============================================================

def rerank_movies(query, movies):
    if not movies:
        return []

    pairs = []
    for m in movies:
        context = f"Title: {m.get('title')}. Genres: {m.get('genres')}. Plot: {m.get('overview')}. Keywoords: {m.get('keywords')}. Cast: {m.get('cast')}. Director: {m.get('director')}"
        pairs.append([query, context])
    
    scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
    
    for movie, score in zip(movies, scores):
        movie["cross_score"] = float(score)
        # Apply your popularity boost here if desired
        pop = movie.get("popularity", 0) or 0
        movie["final_score"] = movie["cross_score"] + (math.log(pop + 1) * 0.85)

    movies.sort(key=lambda x: x["final_score"], reverse=True)
    return movies


# chatgpt/rrf.py

def search_rrf_pipeline(user_query, top_n=150):
    """
    The full high-performance pipeline for evaluation.
    """
    # 1. Rewriting
    rewritten = rewrite_query(user_query)
    
    # 2. Retrieval (Broad candidates)
    # We fetch 50 from each to give RRF and Reranker enough to work with
    bm25_res = get_bm25_results(rewritten, top_n=150)
    vec_res = get_vector_results(rewritten, top_n=150)
    
    # 3. Fusion
    fused = reciprocal_rank_fusion(bm25_res, vec_res, top_n=150)
    
    # 4. Reranking & Popularity Boost
    final_results = rerank_movies(user_query, fused)
    
    return final_results[:top_n]
# ============================================================
# 5. EXECUTION PIPELINE
# ============================================================

if __name__ == "__main__":
    test_query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion."
    test_query = "thriller sci-fi drama movie about two stage magicians engage in competitive one-upmanship in an attempt to create the ultimate stage illusion."
    test_query = "action, epic movies about ancient Greeks their mythology, God`s, battles, voyages, adventures and wars "
    test_query = "thriller drama movie where the husband of a missing woman becomes the main suspect in her disappearance with a Ben Affleck in the main role"
    #test_query = "A drama romance movie about a guy from Alabama with a low IQ who ran in many countries."
    # test_query = "romantic love story aboard of giant ship of 20th‑century ship that sinks after hitting iceberg"
    # test_query = "movie where toys come to life"
    # test_query = "young FBI trainee looking for help of serial cannibal killer"
    #test_query = "movie about how Harvard undergrad student programmer created Facebook"
    test_query = "Weary Wolverine cares for an ailing Professor X in a hideout on the Mexican border"
    test_query = "Faded actor best known for playing a superhero attempts a comeback on Broadway"
    test_query = "The story of Henry Hill and his life in the mob Ray Liotta Robert De Niro"
    test_query = "Farm boy joins a galactic rebellion and learns about the Force"
    test_query = "Scottish warrior leads a group of people against the English king with Mel Gibson main role"
    #test_query = "A former hitman tries to settle down but is pulled back for one last job"
    print(f"\n🔎 Query: {test_query}")

    # Step 1: Rewriting
    rewritten = rewrite_query(test_query)
    print(f"📝 Rewritten: {rewritten}")

    # Step 2: Search
    # bm25_res = get_bm25_results(test_query, top_n=50)
    # vec_res = get_vector_results(rewritten, top_n=50)

    # # Step 3: Fusion
    # fused = reciprocal_rank_fusion(bm25_res, vec_res, top_n=30)
    # print(f"✅ Fused {len(fused)} unique candidates.")

    # Step 4: Reranking
    #final_results = rerank_movies(test_query, fused)
    final_results = search_rrf_pipeline(user_query=test_query, top_n=150)


    # Output
    print("\n" + "="*50)
    print("TOP 5 HYBRID RESULTS (RRF + RERANK)")
    print("="*50)
    for i, m in enumerate(final_results[:30], 1):
        print(f"{i}. {m['title']} (Final: {m['final_score']:.2f} | RRF: {m['rrf_score']:.4f})")
        print(f"   Overview: {m['overview'][:100]}...")
        print("-" * 50)