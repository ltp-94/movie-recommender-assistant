from rag.llm_query_rewriting import rewrite_query
from rag.vector import vector_search
from rag.bm_25 import bm25_search


def reciprocal_rank_fusion(bm25_results, vector_results, k=60, top_n=150):
   scores = {}
   documents = {}
   # BM25
   for rank, hit in enumerate(bm25_results, start=1):
       doc_id = hit["_id"]
       scores[doc_id] = scores.get(doc_id, 0) + (0.6 / (k + rank))
       documents[doc_id] = hit["_source"]
   # Vector
   for rank, hit in enumerate(vector_results, start=1):
       doc_id = hit["_id"]
       scores[doc_id] = scores.get(doc_id, 0) + (0.4 / (k + rank))
       documents[doc_id] = hit["_source"]
   ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
   results = []
   for doc_id, rrf_score in ranked[:top_n]:
       movie = documents[doc_id].copy()
       movie["rrf_score"] = rrf_score
       movie["_id"] = doc_id
       results.append(movie)
   return results



# if __name__ == "__main__":
#     query = "drama movie about magicians engage in competitive in an attempt to create the ultimate stage illusion, directed by Nolan"
#     #query = "A psychological thriller where a man has no short term memory"
#     #query = "Scottish warrior leads a group of people against the English king with Mel Gibson main role"
#     #query = "crime, drama movie where young daughter is disappear with her friend and police fails to find them, Hugh Jackman starring"
#     bm25_results = bm25_search(query, retrieve_k=200)
#     rewritten_query = rewrite_query(query)
#     print(rewritten_query)
#     vector_results = vector_search(rewritten_query, retrieve_k=200)
#     candidates = reciprocal_rank_fusion(bm25_results, vector_results, top_n=200)
#     #final_results = rerank_movies(query, candidates)
#     # for movie in final_results[:30]:
#     #     print(movie["title"])
#     for i, el in enumerate(candidates, start=1):
#         print(f"{i}: {el["title"]}")
#         if el["title"] == "Prisoners":
#             print(i)
#             break

