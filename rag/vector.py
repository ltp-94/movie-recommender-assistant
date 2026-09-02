import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import torch

# Modular imports
from rag.llm_query_rewriting import rewrite_query
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL



load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Initialize Elasticsearch
es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=60, retry_on_timeout=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


def vector_search(query, retrieve_k=100):
   query_for_embedding = (
                            "Represent this sentence for searching relevant movie plots: " + query
                            )
   query_vector = bi_encoder.encode(query_for_embedding, normalize_embeddings=True).tolist()
   body = {
       "size": retrieve_k,
       "knn": {
           "field": "embeddings",
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
   response = es.search(index=INDEX_NAME, body=body)
   return response["hits"]["hits"]



# if __name__ == "__main__":
#     test_query = "A psychological thriller where a man has no short term memory"
#     test_query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion"
    
#     print(f"📡 Testing Vector Search for: '{test_query}'\n")
#     start = time.time()
#     results = vector_search(test_query, retrieve_k=100)
    
#     if not results:
#         print("❌ No results found.")
#     else:
#         for i, movie in enumerate(results, 1):
#             # print(f"{i}. {movie['title']} (Score: {movie['vector_score']:.4f})")
#             # print(f"   Overview: {movie.get('overview', '')[:100]}...")
#             # print("-" * 50)
#             print(f"{i}: {movie["_source"]["title"]}")
            
    
#     print(f"Search completed in {time.time() - start:.2f}s")