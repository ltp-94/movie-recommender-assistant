# Modular imports
from rag.llm_query_rewriting import rewrite_query
#from rag.llm_recommendation import generate_recommendation
from rag.vector import vector_search#, bi_encoder, cross_encoder
from rag.bm_25 import bm25_search
from rag.rrf import reciprocal_rank_fusion
from rag.reranker import rerank_movies


def search_rrf_pipeline(query, top_n=20):
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
    result = search_rrf_pipeline(query, top_n=20)
    for el in result:
        print(el["title"])
    #print(llm_gen)