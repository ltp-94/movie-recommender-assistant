import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import os
import time
import torch
import math

# Modular imports
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
from rag.llm_query_rewriting import rewrite_query
from rag.llm_generation import generate_recommendation
# Import your index name and URL from your config
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL



load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Initialize Elasticsearch
es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=60,
    retry_on_timeout=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


def bm25_search(query, retrieve_k=100):
   body = {
       "size": retrieve_k,
       "query": {
           "multi_match": {
               "query": query,
               "fields": [
                   "genres.text",
                   "overview^2",
                   "cast^3",
                   "director",
                   "search_context^3"
               ],
               "type": "best_fields"
           }
       },
       "_source": [
           "title",
           "overview",
           "genres",
           "director",
           "writers",
           "cast",
           "keywords",
           "popularity",
           "vote_average",
           "poster_url",
           "movie_link",
           "release_year",
           "runtime"
       ]
   }
   response = es.search(
       index=INDEX_NAME,
       body=body
   )
   return response["hits"]["hits"]

   
def vector_search(rewritten_query, retrieve_k=100):
   query_for_embedding = (
       "Represent this sentence for searching relevant movie plots: "
       + rewritten_query
   )
   query_vector = bi_encoder.encode(
       query_for_embedding,
       normalize_embeddings=True
   ).tolist()
   body = {
       "size": retrieve_k,
       "knn": {
           "field": "embedding",
           "query_vector": query_vector,
           "k": retrieve_k,
           "num_candidates": 200
       },
       "_source": [
           "title",
           "overview",
           "genres",
           "director",
           "writers",
           "cast",
           "keywords",
           "popularity",
           "vote_average",
           "poster_url",
           "movie_link",
           "release_year",
           "runtime"
       ]
   }
   response = es.search(
       index=INDEX_NAME,
       body=body
   )
   return response["hits"]["hits"]



def reciprocal_rank_fusion(
   bm25_results,
   vector_results,
   k=60,
   top_n=150
):
   scores = {}
   documents = {}
   # BM25
   for rank, hit in enumerate(bm25_results, start=1):
       doc_id = hit["_id"]
       scores[doc_id] = scores.get(doc_id, 0) + (
           0.4 / (k + rank)
       )
       documents[doc_id] = hit["_source"]
   # Vector
   for rank, hit in enumerate(vector_results, start=1):
       doc_id = hit["_id"]
       scores[doc_id] = scores.get(doc_id, 0) + (
           0.3 / (k + rank)
       )

       documents[doc_id] = hit["_source"]
   ranked = sorted(
       scores.items(),
       key=lambda x: x[1],
       reverse=True
   )
   results = []
   for doc_id, rrf_score in ranked[:top_n]:
       movie = documents[doc_id].copy()
       movie["rrf_score"] = rrf_score
       movie["_id"] = doc_id
       results.append(movie)
   return results



# bm25_results = bm25_search(
#    query,
#    retrieve_k=100
# )
# vector_results = vector_search(
#    rewritten_query,
#    retrieve_k=100
# )
# candidates = reciprocal_rank_fusion(
#    bm25_results,
#    vector_results,
#    top_n=100
# )



def rerank_movies(query, movies):
   pairs = []
   for m in movies:
        context = f"Title: {m.get('title')}. Genres: {m.get('genres')}. Plot: {m.get('overview')}. Keywords: {m.get('keywords')}. Cast: {m.get('cast')}. Director: {m.get('director')}"
        pairs.append([query, context])
    
   scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
#    for movie, score in zip(movies, scores):
#        movie["cross_score"] = float(score)
   for movie, score in zip(movies, scores):
        movie["cross_score"] = float(score)
        # Apply your popularity boost here if desired
        pop = movie.get("popularity", 0) or 0
        movie["final_score"] = movie["cross_score"] + (math.log(pop + 1) * 0.85)

   movies.sort(key=lambda x: x["final_score"], reverse=True) # Sorts by raw AI score only
   return movies


# Inside rrf_pipeline.py

def search_rrf_evaluation_pipeline(query, top_n=20):
    """
    Unified function for the evaluation script to call.
    """

    rewritten = rewrite_query(query)
    print(rewritten)
    # 1. BM25 Search
    bm25_results = bm25_search(rewritten, retrieve_k=150)
    
    # 2. Query Rewriting
#    rewritten = rewrite_query(query)
    
    # 3. Vector Search
    vector_results = vector_search(rewritten, retrieve_k=150)
    
    # 4. Fusion
    candidates = reciprocal_rank_fusion(bm25_results, vector_results, top_n=50)
    
    # 5. Reranking (This is the slow part)
    # We only rerank 50 to keep the evaluation from taking hours
    final_results = rerank_movies(query, candidates)
    
    return final_results[:top_n]


if __name__ == "__main__":
    query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion, directed by Nolan"
    #query = "A psychological thriller where a man has no short term memory"
    #query = "Scottish warrior leads a group of people against the English king with Mel Gibson main role"
    query = "crime, drama movie where young daughter is disappear with her friend and police fails to find them, Hugh Jackman starring"
    # bm25_results = bm25_search(query, retrieve_k=150)
    # rewritten_query = rewrite_query(query)
    # print(rewritten_query)
    # vector_results = vector_search(
    #                             rewritten_query,
    #                             retrieve_k=150
    #                             )
    # candidates = reciprocal_rank_fusion(
    #                             bm25_results,
    #                             vector_results,
    #                             top_n=100
    #                             )
    # final_results = rerank_movies(query, candidates)
    # for movie in final_results[:30]:
    #     print(movie["title"])
    result = search_rrf_evaluation_pipeline(query, top_n=20)
    llm_gen = generate_recommendation(query, result)
    for el in result:
        print(el["title"])

