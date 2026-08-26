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




# To this:
es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=60,       # Increased to 60 seconds
    retry_on_timeout=True, 
    max_retries=3
)



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


# if __name__ == "__main__":
#     # test_query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion."
#     # test_query = "thriller sci-fi drama movie about two stage magicians engage in competitive one-upmanship in an attempt to create the ultimate stage illusion."
#     # test_query = "action, epic movies about ancient Greeks their mythology, God`s, battles, voyages, adventures and wars "
#     # test_query = "thriller drama movie where the husband of a missing woman becomes the main suspect in her disappearance with a Ben Affleck in the main role"
#     #test_query = "A drama romance movie about a guy from Alabama with a low IQ who ran in many countries."
#     # test_query = "romantic love story aboard of giant ship of 20th‑century ship that sinks after hitting iceberg"
#     # test_query = "movie where toys come to life"
#     # test_query = "young FBI trainee looking for help of serial cannibal killer"
#     #test_query = "movie about how Harvard undergrad student programmer created Facebook"
#     # test_query = "Weary Wolverine cares for an ailing Professor X in a hideout on the Mexican border"
#     # test_query = "Faded actor best known for playing a superhero attempts a comeback on Broadway"
#     # test_query = "The story of Henry Hill and his life in the mob Ray Liotta Robert De Niro"
#     # test_query = "Farm boy joins a galactic rebellion and learns about the Force"
#     # test_query = "Scottish warrior leads a group of people against the English king with Mel Gibson main role"
#     test_query = "crime, drama movie where young daughter is disappear with her friend and police fails to find them, Hugh Jackman starring"
#     #test_query = "A former hitman tries to settle down but is pulled back for one last job"
#     print(f"\n🔎 Query: {test_query}")

#     # Step 1: Rewriting
#     rewritten = rewrite_query(test_query)
#     print(f"📝 Rewritten: {rewritten}")
#     result = get_vector_results(rewritten, top_n=100)
#     for m in result:
#         print(m["content"]['title'])


