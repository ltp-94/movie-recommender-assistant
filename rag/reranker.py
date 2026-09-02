from ingestion.config import INDEX_NAME, ELASTICSEARCH_URL
from rag.llm_query_rewriting import rewrite_query
from rag.vector import vector_search, bi_encoder, cross_encoder
from rag.bm_25 import bm25_search
from rag.rrf import reciprocal_rank_fusion
import numpy as np


# # To this:
# es = Elasticsearch(
#     ELASTICSEARCH_URL,
#     request_timeout=60,       # Increased to 60 seconds
#     retry_on_timeout=True, 
#     max_retries=3
# )



def rerank_movies(query, movies):
    if not movies:
        return []

    pairs = []
    for m in movies:
        context = f"Title: {m.get('title')}. Genres: {m.get('genres')}. Plot: {m.get('overview')}.\
                    Keywords: {m.get('keywords')}. Cast: {m.get('cast')}. Director: {m.get('director')}"
        pairs.append([query, context])
    
    scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
    
    for movie, score in zip(movies, scores):
        movie["cross_score"] = float(score)
        # Apply your popularity boost here if desired
        pop = movie.get("popularity", 0) or 0
        movie["final_score"] = movie["cross_score"] + (np.log(pop + 1) * 0.85)

    movies.sort(key=lambda x: x["final_score"], reverse=True)
    return movies



# if __name__ == "__main__":
#     query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion, directed by Nolan"
#     #query = "A psychological thriller where a man has no short term memory"
#     #query = "Scottish warrior leads a group of people against the English king with Mel Gibson main role"
#     query = "crime, drama movie where young daughter is disappear with her friend and police fails to find them, Hugh Jackman starring"
#     bm25_results = bm25_search(query, retrieve_k=150)
#     rewritten_query = rewrite_query(query)
#     print(rewritten_query)
#     vector_results = vector_search(
#                                 rewritten_query,
#                                 retrieve_k=150
#                                 )
#     candidates = reciprocal_rank_fusion(
#                                 bm25_results,
#                                 vector_results,
#                                 top_n=100
#                                 )
#     final_results = rerank_movies(query, candidates)

#     for movie in final_results[:30]:
#         print(movie["title"])