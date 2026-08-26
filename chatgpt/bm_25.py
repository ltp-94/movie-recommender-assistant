import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Import your index name and URL from your config
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL

load_dotenv()

# Initialize Elasticsearch
es = Elasticsearch(
    ELASTICSEARCH_URL,
    request_timeout=60,
    retry_on_timeout=True
)

# def bm25_search(query, top_n=10):
#     """
#     Performs a standard keyword search (BM25) using multi_match.
#     """
#     search_body = {
#         "size": top_n,
#         "query": {
#             "multi_match": {
#                 "query": query,
#                 "fields": [
#                     "genres.text^3",
#                     "overview^2",   # Boost overview relevance
#                     "keywords",
#                     "cast^2",       # High boost for actor names
#                     "director",
#                     "search_context^3" # High boost for combined context
#                 ],
#                 "type": "best_fields"
#             }
#         },
#         "_source": [
#             "title", "overview", "genres", "director", "writers", 
#             "cast", "keywords", "popularity", "vote_average", 
#             "poster_url", "movie_link", "release_year", "runtime"
#         ]
#     }

#     response = es.search(index=INDEX_NAME, body=search_body)
    
#     # Extract just the source data and the score
#     hits = []
#     for hit in response["hits"]["hits"]:
#         doc = hit["_source"]
#         doc["bm25_score"] = hit["_score"] # Elasticsearch's BM25 score
#         hits.append(doc)
        
#     return hits


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
# ============================================================
# RUN TEST
# ============================================================
# if __name__ == "__main__":
#     test_query = "thriller movie with Ben Affleck"
#     #test_query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion"
    
#     print(f"🔎 Testing BM25 Search for: '{test_query}'\n")
    
#     results = get_bm25_results(test_query, top_n=15)
    
#     if not results:
#         print("❌ No results found.")
#     else:
#         # for i, movie in enumerate(results, 1):
#         #     print(f"{i}. {movie['title']} (Score: {movie['bm25_score']:.2f})")
#         #     print(f"   Year: {movie.get('release_year', 'N/A')} | Director: {movie.get('director', 'N/A')}")
#         #     print(f"   Overview: {movie.get('overview', '')[:100]}...")
#         #     print("-" * 50)
#         for m in results:
#             print(m["content"]["title"])


        