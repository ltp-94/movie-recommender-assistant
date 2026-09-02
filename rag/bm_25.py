from elasticsearch import Elasticsearch
from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL


# Initialize Elasticsearch
es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=60, retry_on_timeout=True)


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
   response = es.search(index=INDEX_NAME, body=body)
   return response["hits"]["hits"]



# if __name__ == "__main__":
#     test_query = "thriller movie with Ben Affleck"
#     test_query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion"
    
#     print(f"🔎 Testing BM25 Search for: '{test_query}'\n")
    
#     results = bm25_search(test_query)
    
#     if not results:
#         print("❌ No results found.")
#     else:
#         for i, movie in enumerate(results, 1):
#             # print(f"{i}. {movie['title']} (Score: {movie['bm25_score']:.2f})")
#             # print(f"   Year: {movie.get('release_year', 'N/A')} | Director: {movie.get('director', 'N/A')}")
#             # print(f"   Overview: {movie.get('overview', '')[:100]}...")
#             # print("-" * 50)
#             print(f"{i}: {movie["_source"]["title"]}")
