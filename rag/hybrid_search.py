# import time
# import torch
# import os
# from openai import OpenAI
# from elasticsearch import Elasticsearch
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from dotenv import load_dotenv

# # Modular imports
# from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
# from rag.llm_query_rewriting import rewrite_query

# load_dotenv()

# # ============================================================
# # SETUP
# # ============================================================
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Set offline mode if models are already downloaded
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# es = Elasticsearch(ELASTICSEARCH_URL)

# client = OpenAI(
#     api_key=os.getenv("GROQ_API_KEY"), 
#     base_url="https://api.groq.com/openai/v1"
# )

# # ============================================================
# # HYBRID SEARCH
# # ============================================================

# def search_movies(query_text, top_n=10, retrieve_k=100):
#     start_time = time.time()
    
#     # 1. Rewrite query for better retrieval
#     rewritten_query = rewrite_query(query_text)
    
#     # 2. Encode query
#     query_for_embedding = f"Represent this sentence for searching relevant movie plots: {rewritten_query}"
#     query_vector = bi_encoder.encode(query_for_embedding, normalize_embeddings=True).tolist()

#     # 3. Search Body (Keeping your specific boosts and fields)
#     search_body = {
#         "size": retrieve_k,
#         "query": {
#             "bool": {
#                 "should": [
#                     {
#                         "multi_match": {
#                             "query": rewritten_query,
#                             "fields": [
#                                 "genres.text", 
#                                 "overview^2", 
#                                 "cast^3", 
#                                 "director", 
#                                 "writers", 
#                                 "keywords",        # Added back for precision
#                                 "search_context^3"
#                             ],
#                             "type": "best_fields",
#                             "boost": 0.85
#                         }
#                     }
#                 ]
#             }
#         },
#         "knn": {
#             "field": "embedding",
#             "query_vector": query_vector,
#             "k": retrieve_k,
#             "num_candidates": 300,
#             "boost": 3.5
#         },
#         "_source": [
#             "title", "overview", "genres", "director", "writers", 
#             "cast", "keywords", "poster_url", "movie_link", 
#             "runtime", "release_year", "vote_average"
#         ]
#     }

#     response = es.search(index=INDEX_NAME, body=search_body)
#     hits = []
#     for hit in response["hits"]["hits"]:
#         doc = hit["_source"]
#         doc["es_score"] = hit["_score"]
#         hits.append(doc)

#     if not hits:
#         return []

#     # ============================================================
#     # PRECISE RERANKING (The logic you liked)
#     # ============================================================
#     pairs = []
#     for movie in hits:
#         # We keep the detailed multiline context that makes the Cross-Encoder smarter
#         context = f"""
# Title: {movie.get('title','')}
# Genres: {movie.get('genres','')}
# Director: {movie.get('director','')}
# Writers: {movie.get('writers','')}
# Cast: {movie.get('cast','')}
# Keywords: {movie.get('keywords','')}
# Overview: {movie.get('overview','')}
# """
#         pairs.append([rewritten_query, context])

#     scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)

#     for movie, score in zip(hits, scores):
#         movie["rerank_score"] = float(score)

#     hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    
#     print(f"🔍 Search completed in {time.time() - start_time:.2f}s")
#     return hits[:top_n]

# # ============================================================
# # GENERATION (RAG)
# # ============================================================

# def generate_recommendation(user_query, reranked_results):
#     context = ""
#     for movie in reranked_results[:8]: # Top 8 for a balanced prompt
#         context += f"Title: {movie['title']} ({movie.get('release_year', 'N/A')})\n"
#         context += f"Rating: {movie.get('vote_average', 'N/A')}/10\n"
#         context += f"Director: {movie['director']}\n"
#         context += f"Cast: {movie['cast']}\n"
#         context += f"Plot: {movie['overview']}\n"
#         context += f"Keywords: {movie['keywords']}\n\n"

#     system_prompt = """
#     You are a professional movie critic and recommendation engine.
#     Use the provided movie context to answer the user's request.
#     Explain WHY you are recommending these specific movies based on their plot, cast, and year.
#     Always provide a helpful and conversational response.
#     """

#     user_prompt = f"User Request: {user_query}\n\nMovie Context:\n{context}"

#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile", # Updated for current Groq support
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0.7
#     )

#     return response.choices[0].message.content

# # ============================================================
# # ENTRY POINT
# # ============================================================
# if __name__ == "__main__":
#     query = "action, epic movies about ancient Greeks their mythology, battles and voyages"
#     query = "story about how student programmer created Facebook and obstacles that followed"
#     results = search_movies(query, top_n=30)

#     print(f"\nQUERY: {query}")
#     print("\n" + "="*30 + "\nAI RECOMMENDATION\n" + "="*30)
#     print(generate_recommendation(query, results))
#     for el in results:
#         print(el["title"])

from sentence_transformers import SentenceTransformer, CrossEncoder
from elasticsearch import Elasticsearch
import torch
import time
from openai import OpenAI
import os
from llm_query_rewriting import rewrite_query



# ============================================================
# SETUP
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"



# Load from local directory instead of HF Hub
bi_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
#cross_encoder = CrossEncoder("/workspaces/movie_recommendation/models/hf_models/cross-encoder-ms-marco-TinyBERT-L-2-v2", device=device)

es = Elasticsearch(
   "http://localhost:9200",
   request_timeout=60,
   retry_on_timeout=True
)

INDEX_NAME = "movies"


# ============================================================
# HYBRID SEARCH
# ============================================================

def search_movies(query_text, top_n=10, retrieve_k=150):
   start_time = time.time()
   # BGE performs better with an instruction
   query_text = rewrite_query(query_text)
   print(query_text)
   query_for_embedding = (
       "Represent this sentence for searching relevant movie plots: "
       + query_text
   )

   query_vector = bi_encoder.encode(
       query_for_embedding,
       normalize_embeddings=True
   ).tolist()

   search_body = {

       "size": retrieve_k,

       "query": {

           "bool": {

               "should": [

                   {
                       "multi_match": {

                           "query": query_text,

                           "fields": [

                               #"keywords",

                               "genres.text",

                               "overview^2",

                               "cast^3",

                               "director",

                               "writers",

                               "search_context^3"

                           ],

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

           "title",

           "overview",

           "genres",

           "director",

           "writers",

           "cast",

           "keywords"

       ]
   }

   response = es.search(
       index=INDEX_NAME,
       body=search_body
   )

   hits = []

   for hit in response["hits"]["hits"]:

       doc = hit["_source"]

       doc["es_score"] = hit["_score"]

       hits.append(doc)

   if not hits:
       return []

   # ============================================================
   # CROSS ENCODER
   # ============================================================

   pairs = []

   for movie in hits:

       context = f"""
Title: {movie.get('title','')}

Genres: {movie.get('genres','')}

Director: {movie.get('director','')}

Writers: {movie.get('writers','')}

Cast: {movie.get('cast','')}

Keywords: {movie.get('keywords','')}

Overview:
{movie.get('overview','')}
"""

       pairs.append([query_text, context])

   scores = cross_encoder.predict(
       pairs,
       batch_size=8,
       show_progress_bar=True
   )

   for movie, score in zip(hits, scores):
       movie["rerank_score"] = float(score)

   hits.sort(
       key=lambda x: x["rerank_score"],
       reverse=True
   )
   end_time = time.time()
   print(f"Total search_movies execution time: {end_time - start_time:.2f} seconds")
   return hits[:top_n]



def search_vector_only(query_text, top_n=5):
    # Just the KNN part, no multi_match, no re-ranking
    query_vector = bi_encoder.encode(query_text, normalize_embeddings=True).tolist()
    search_body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_n,
            "num_candidates": 100
        }
    }
    response = es.search(index="movies", body=search_body)
    return [hit["_source"] for hit in response["hits"]["hits"]]


# ... (all your imports and search_movies function above)

# if __name__ == "__main__":
#     # This code ONLY runs when you execute 'python hybrid_search.py'
#     # It will NOT run when you import it into evaluation.py
#     query = "thriller movie with Ben Affleck as a husband whose wife disappears"
#     results = search_movies(query, top_n=10)
#     print(f"Top result: {results[0]['title']}")

# ============================================================
# EXAMPLE
# ============================================================

query = "thriller sci-fi drama movie about two stage magicians engage in competitive one-upmanship in an attempt to create the ultimate stage illusion."
query = "action, epic movies about ancient Greeks their mythology, God`s, battles, voyages, adventures and wars "
#query = "thriller drama movie where the husband of a missing woman becomes the main suspect in her disappearance with a Ben Affleck in the main role"
#query = "A drama romance movie about a guy from Alabama with a low IQ who ran in many countries."
# query = "romantic love story aboard of giant ship of 20th‑century ship that sinks after hitting iceberg"
# query = "movie where toys come to life"
# query = "young FBI trainee looking for help of serial cannibal killer"
query = "movie about how Harvard undergrad student programmer created Facebook"
# query = "Weary Wolverine cares for an ailing Professor X in a hideout on the Mexican border"

results = search_movies(query, top_n=30)

print(f"\nQuery: {query}\n")

for i, movie in enumerate(results, start=1):

   print(f"{i}. {movie['title']}")

   print(f"CrossEncoder: {movie['rerank_score']:.3f}")

   print(f"ES Score: {movie['es_score']:.3f}")

   print(movie["overview"][:120])

   print("-" * 60)





def generate_recommendation(user_query, reranked_results):
    # 1. Prepare the context from your top Reranked results
    client = OpenAI(api_key="os.getenv("GROQ_API_KEY")",  base_url="https://api.groq.com/openai/v1")

    context = ""
    for movie in reranked_results:#[:20]: # Use top 5 for best LLM performance
        context += f"Title: {movie['title']}\n"
        context += f"Director: {movie['director']}\n"
        context += f"Writers: {movie['writers']}\n"
        context += f"Cast: {movie['cast']}\n"
        context += f"Plot: {movie['overview']}\n"
        context += f"Keywords: {movie['keywords']}\n"

    system_prompt = """
    You are a professional movie critic and recommendation engine.
    Use the provided movie context to answer the user's request.
    Explain WHY you are recommending these specific movies based on their plot and cast.
    If the context doesn't contain a relevant movie, say you don't know.
    """

    user_prompt = f"User Request: {user_query}\n\nMovie Context:\n{context}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# --- FULL FLOW ---
#query = "thriller movie with Ben Affleck as a husband whose wife disappears"

# 1. Retrieve & Rerank (from your previous Elasticsearch code)
retrieved_results = search_movies(query, top_n=10)

# 2. Generate (Retrieval-Augmented Generation)
ai_answer = generate_recommendation(query, retrieved_results)

print("RECOMMENDATION:\n", ai_answer)

