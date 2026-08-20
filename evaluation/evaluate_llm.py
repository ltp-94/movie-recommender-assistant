# import pandas as pd
# import numpy as np
# #from hybrid_search import search_movies, search_vector_only
# from rag.search_popularity import search_movies

# def calculate_metrics(ground_truth_path, top_n=5):
#     df_gt = pd.read_csv(ground_truth_path)
#     total = len(df_gt)
    
#     hits = 0
#     reciprocal_ranks = []
    
#     print(f"🚀 Starting Evaluation on {total} queries (Top-{top_n})...\n")
    
#     for index, row in df_gt.iterrows():
#         query = row['query']
#         expected_title = row['expected_title']
        
#         # 1. Run Search
#         results = search_movies(query, top_n=top_n)
#         top_titles = [m['title'] for m in results]
        
#         # 2. Calculate Hit Rate & MRR
#         if expected_title in top_titles:
#             hits += 1
#             # Find the 1-based rank (1, 2, 3...)
#             rank = top_titles.index(expected_title) + 1
#             reciprocal_ranks.append(1 / rank)
#             print(f"✅ PASS | Rank {rank} | {expected_title}")
#         else:
#             reciprocal_ranks.append(0)
#             print(f"❌ FAIL | Not found | Expected: {expected_title}")

#     # Final calculations
#     hit_rate = hits / total
#     mrr = np.mean(reciprocal_ranks)
    
#     return hit_rate, mrr

# if __name__ == "__main__":
#     # Note: Using your absolute path from the error log
#     GT_PATH = "/workspaces/movie-recommender-assistant/data/ground_truth.csv"
    
#     hr, mrr = calculate_metrics(GT_PATH, top_n=30)
    
#     print("\n" + "="*40)
#     print(f"RETRIEVAL PERFORMANCE (K=5)")
#     print("-" * 40)
#     print(f"HIT RATE: {hr:.2%}")
#     print(f"MRR:      {mrr:.3f}")
#     print("="*40)

import polars as pl
import time
#from rag.search_popularity import search_movies
from chatgpt.rrf import search_rrf_pipeline as search_movies 



def calculate_metrics(ground_truth_path: str, top_n: int = 5):
    # 1. Load Ground Truth with Polars
    df_gt = pl.read_csv(ground_truth_path)
    total = len(df_gt)
    
    # We will store results here to calculate metrics at the end
    results_data = []
    
    print(f"🚀 Starting Evaluation on {total} queries (Top-{top_n})...\n")
    start_eval_time = time.time()

    # 2. Iterate through rows
    # iter_rows(named=True) gives us a dictionary for each row
    for row in df_gt.iter_rows(named=True):
        query = row['query']
        expected_title = row['expected_title']
        
        # Run Search logic (Elastic + Rerank + Popularity)
        search_results = search_movies(query, top_n=top_n)
        top_titles = [m['title'] for m in search_results]
        
        # Calculate Rank
        # If expected_title matches any of the top_titles
        if expected_title in top_titles:
            # .index is 0-based, so we add 1 for the actual rank
            rank = top_titles.index(expected_title) + 1
            reciprocal_rank = 1 / rank
            print(f"✅ PASS | Rank {rank:2} | {expected_title}")
        else:
            rank = None
            reciprocal_rank = 0.0
            print(f"❌ FAIL | Not found | Expected: {expected_title}")
            
        results_data.append({
            "query": query,
            "expected": expected_title,
            "rank": rank,
            "rr": reciprocal_rank
        })

    # 3. Use Polars to calculate final metrics
    results_df = pl.DataFrame(results_data)
    
    # Hit Rate: Total successful hits / total queries
    hit_rate = results_df.filter(pl.col("rank").is_not_null()).height / total
    
    # MRR: Mean of Reciprocal Ranks
    mrr = results_df["rr"].mean()
    
    duration = time.time() - start_eval_time
    print(f"\nEvaluation finished in {duration:.2f}s")
    
    return hit_rate, mrr

if __name__ == "__main__":
    GT_PATH = "/workspaces/movie-recommender-assistant/data/ground_truth.csv"
    
    # You can set top_n=10 or 30 depending on how strict you want to be
    TOP_K = 20
    hr, mrr = calculate_metrics(GT_PATH, top_n=TOP_K)
    
    print("\n" + "="*40)
    print(f"RETRIEVAL PERFORMANCE (K={TOP_K})")
    print("-" * 40)
    print(f"HIT RATE: {hr:.2%}")
    print(f"MRR:      {mrr:.3f}")
    print("="*40)