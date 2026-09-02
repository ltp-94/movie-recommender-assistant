import os
import time
import math
import torch
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Modular imports
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
from rag.llm_query_rewriting import rewrite_query
from rag.llm_generation import generate_recommendation

load_dotenv()

# ============================================================
# SETUP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load from local directory as per your setup
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


es = Elasticsearch(
   ELASTICSEARCH_URL,
   request_timeout=60,
   retry_on_timeout=True
)

# client = OpenAI(
#     api_key=os.getenv("GROQ_API_KEY"), 
#     base_url="https://api.groq.com/openai/v1"
# )

# ============================================================
# HYBRID SEARCH WITH POPULARITY BOOST
# ============================================================

def search_movies(query_text, top_n=10, retrieve_k=100):
    start_time = time.time()
    
    # 1. Query Rewriting
    rewritten_query = rewrite_query(query_text)

    print(f"Rewritten Query: {rewritten_query}")

    # 2. Vector Encoding
    query_for_embedding = f"Represent this sentence for searching relevant movie plots: {rewritten_query}"
    query_vector = bi_encoder.encode(query_for_embedding, normalize_embeddings=True).tolist()

    # 3. Hybrid Search Body
    search_body = {
        "size": retrieve_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": rewritten_query,
                            "fields": ["genres.text", "overview^2", "cast^3", "director", "search_context^3"],
                            "type": "best_fields",
                            "boost": 0.85
                        }
                    }
                ]
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": retrieve_k,
            "num_candidates": 300,
            "boost": 3.5
        },
        "_source": [
            "title", "overview", "genres", "director", "writers", "cast", 
            "keywords", "popularity", "vote_average", "poster_url", 
            "movie_link", "release_year", "runtime"
        ]
    }

    response = es.search(index=INDEX_NAME, body=search_body)
    hits = []
    for hit in response["hits"]["hits"]:
        doc = hit["_source"]
        doc["es_score"] = hit["_score"]
        hits.append(doc)

    if not hits:
        return []

    # 4. Cross-Encoder Reranking
    pairs = []
    for movie in hits:
        # PRECISE CONTEXT: Using the specific format that gave you better results
        context = (
            f"Title: {movie.get('title','')}. "
            f"Genres: {movie.get('genres','')}. "
            f"Plot: {movie.get('overview','')}. "
            f"Keywords: {movie.get('keywords','')}."
        )
        pairs.append([query_text, context])

    scores = cross_encoder.predict(pairs, batch_size=8, show_progress_bar=False)

    # 5. Apply Popularity Boost (Log Scaling)
    # Weight of 0.56 as per your specific preference
    for movie, score in zip(hits, scores):
        popularity = movie.get('popularity', 0) or 0
        pop_boost = math.log(popularity + 1) * 3.5
        movie["rerank_score"] = float(score) + pop_boost

    # Sort by the final boosted score
    hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    print(f"🔍 Search & Boost completed in {time.time() - start_time:.2f}s")
    return hits[:top_n]

# ============================================================
# LLM GENERATION (RAG)
# ============================================================

# def generate_recommendation(user_query, reranked_results):
#     """
#     Takes the top boosted results and generates a conversational AI response.
#     """
#     context = ""
#     for movie in reranked_results[:10]:
#         context += f"Title: {movie['title']} ({movie.get('release_year', 'N/A')})\n"
#         context += f"Genres: {movie.get('genres', 'N/A')}\n"
#         context += f"Director: {movie.get('director', 'N/A')}\n"
#         context += f"Plot: {movie.get('overview', 'N/A')}\n\n"

#     system_prompt = """
#         You are a professional movie critic. 
#         IMPORTANT: You may ONLY recommend movies that are provided in the "Movie Context" below.
#         If a movie is not in the context, DO NOT mention it.
#         Explain WHY you are recommending these specific movies based ONLY on the provided plot and cast.
#         If no relevant movies are in the context, say you don't know.
#     """

#     user_prompt = f"User Request: {user_query}\n\nMovie Context:\n{context}"

#     response = client.chat.completions.create(
#         model="openai/gpt-oss-20b", # Updated for Groq decommissioning
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.7
#     )

#     return response.choices[0].message.content

# ============================================================
# TEST EXECUTION
# ============================================================
if __name__ == "__main__":
    query = "crime, drama movie where young daughter is disappear with her friend and police fails to find them"
    #query = "movie about how student programmer created Facebook"
    query = "young FBI trainee looking for help of serial cannibal killer"
    query = "romantic love story aboard of giant ship of 20th‑century ship that sinks after hitting iceberg"
    query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion."
    results = search_movies(query, top_n=20)
    
    print(f"\nQUERY: {query}\n")
    for i, m in enumerate(results, 1):
        print(f"{i}. {m['title']} (Boosted Score: {m['rerank_score']:.2f})")
    
    print("\n" + "="*30 + "\nAI RECOMMENDATION\n" + "="*30)
    print(generate_recommendation(query, results))