import polars as pl
import time
#from rag.search_popularity import search_movies
#from chatgpt.rrf import search_rrf_pipeline as search_movies 
from chatgpt.test import search_rrf_evaluation_pipeline as search_movies 



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