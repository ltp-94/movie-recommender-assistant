import os
import json
import polars as pl
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from chatgpt.test import search_rrf_evaluation_pipeline as search_movies 

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Using Llama 3 or similar via Groq
JUDGE_MODEL = "openai/gpt-oss-120b"

# ============================================================
# IMPROVED JSON PROMPT
# ============================================================
JUDGE_PROMPT = """
You are a search quality judge. Evaluate the relevance of the TOP 3 search results for a movie query.

USER QUERY: {query}
EXPECTED MOVIE: {expected}

RESULTS:
{results_text}

SCORING RUBRIC:
3: EXCELLENT - Perfect match or exactly what was requested.
2: GOOD - Highly relevant, same vibe/genre/actors.
1: PARTIAL - Slight connection (e.g., same director but wrong movie).
0: IRRELEVANT - No connection.

You MUST return a JSON object with this structure:
{{
  "evaluations": [
    {{ "rank": 1, "score": number, "reason": "string" }},
    {{ "rank": 2, "score": number, "reason": "string" }},
    {{ "rank": 3, "score": number, "reason": "string" }}
  ]
}}
"""

def get_top_n_judgments(query, expected, movies):
    """
    Evaluates a list of movies in one single LLM call.
    """
    # Format the movies into a string for the prompt
    results_text = ""
    for i, m in enumerate(movies, 1):
        results_text += f"""
                            RANK {i}
                            Title: {m.get('title', '')}
                            Genres: {m.get('genres', '')}
                            Director: {m.get('director', '')}
                            Cast: {m.get('cast', '')}
                            Overview: {m.get('overview', '')}
                            Keywords: {m.get('keywords', '')}
                        """

    prompt = JUDGE_PROMPT.format(
        query=query,
        expected=expected,
        results_text=results_text
    )
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, # Forces JSON output
            temperature=0
        )
        # Parse the JSON response
        data = json.loads(response.choices[0].message.content)
        return data.get("evaluations", [])
    except Exception as e:
        print(f"Error in LLM Judge: {e}")
        return []

def run_judge_evaluation(gt_path, output_path, limit=None):
    df = pl.read_csv(gt_path)
    if limit:
        df = df.head(limit)

    all_results = []
    print(f"🚀 Starting Top-3 Judge Evaluation on {len(df)} samples...\n")

    for row in tqdm(df.iter_rows(named=True)):
        query = row['query']
        expected = row['expected_title']
        
        # 1. Search for Top 3
        search_hits = search_movies(query, top_n=3)
        
        if not search_hits:
            continue
            
        # 2. Get evaluation for all 3 at once
        evaluations = get_top_n_judgments(query, expected, search_hits)
        
        # 3. Combine search data with judge scores
        for i, m in enumerate(search_hits):
            # Find matching judge score by rank
            eval_data = next((item for item in evaluations if item["rank"] == i+1), {"score": 0, "reason": "N/A"})
            
            all_results.append({
                "query": query,
                "expected": expected,
                "retrieved_title": m['title'],
                "rank": i + 1,
                "llm_score": eval_data["score"],
                "llm_reason": eval_data["reason"]
            })

    # 4. Final Metrics
    results_df = pl.DataFrame(all_results)
    
    # Calculate Mean Score @ Rank 1 vs Mean Score @ Rank 3
    avg_rank1 = results_df.filter(pl.col("rank") == 1)["llm_score"].mean()
    avg_overall = results_df["llm_score"].mean()
    
    # "Hit Rate" based on LLM (Is there a score 3 in the top 3?)
    perfect_hits = results_df.filter(pl.col("llm_score") == 3).select("query").unique().height
    hit_rate_at_3 = (perfect_hits / len(df)) * 100

    print(f"\n" + "="*40)
    print(f"TOP-3 JUDGE RESULTS")
    print("-" * 40)
    print(f"Avg Score @ Rank 1: {avg_rank1:.2f}")
    print(f"Avg Score (All Top 3): {avg_overall:.2f}")
    print(f"LLM-Verified Hit Rate @ 3: {hit_rate_at_3:.1f}%")
    print("="*40)

    results_df.write_csv(output_path)

if __name__ == "__main__":
    run_judge_evaluation("../data/ground_truth_10.csv", "/workspaces/movie-recommender-assistant/data/judge_top3_results.csv")