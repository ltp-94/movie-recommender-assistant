import os
import polars as pl
import time
from openai import OpenAI
from dotenv import load_dotenv
from rag.search_popularity import search_movies # Import your search logic
#from config import DATA_PATH

load_dotenv()

# Setup Groq Client for the Judge
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Use a fast but capable model as the Judge
JUDGE_MODEL = "llama-3.1-8b-instant" 

# ============================================================
# THE JUDGE'S RUBRIC
# ============================================================
JUDGE_PROMPT = """
You are an expert movie librarian and search quality judge.
Analyze the relevance of a search result based on a user's query.

USER QUERY: {query}
SEARCH RESULT TITLE: {title}
SEARCH RESULT OVERVIEW: {overview}

SCORING RULES:
3: EXCELLENT - This is exactly what the user wanted (e.g., they asked for a specific plot and you found it).
2: GOOD - Very relevant, fits the genre/theme, but might not be the exact specific movie if one was implied.
1: PARTIAL - Some overlap (e.g., same actor or same genre) but misses the main point of the query.
0: IRRELEVANT - No connection to the query at all.

Provide your output in exactly this format:
Score: [number]
Reason: [one brief sentence explaining the score]
"""

def get_llm_judgment(query, movie):
    """
    Sends a query and a single movie result to the LLM to get a relevance score.
    """
    prompt = JUDGE_PROMPT.format(
        query=query,
        title=movie['title'],
        overview=movie['overview']
    )
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        
        # Parse the score from the "Score: X" line
        score_line = [line for line in content.split('\n') if "Score:" in line][0]
        score = int(score_line.split(':')[1].strip())
        
        reason_line = [line for line in content.split('\n') if "Reason:" in line][0]
        reason = reason_line.split(':')[1].strip()
        
        return score, reason
    except Exception as e:
        return 0, f"Error: {str(e)}"

def run_judge_evaluation(gt_path, output_path, limit=None):
    # 1. Load Ground Truth
    df = pl.read_csv(gt_path)
    if limit:
        df = df.head(limit) # LLM Judge can be slow/expensive, so allow limits

    results = []
    print(f"🚀 Starting LLM Judge Evaluation on {len(df)} samples...\n")

    for row in df.iter_rows(named=True):
        query = row['query']
        expected = row['expected_title']
        
        # 2. Get Top 1 Result from your RAG system
        search_hits = search_movies(query, top_n=1)
        
        if not search_hits:
            results.append({
                "query": query, "expected": expected, "result": "N/A", 
                "score": 0, "reason": "No results found by search engine."
            })
            print(f"❌ FAIL | No result for: {query[:40]}...")
            continue
            
        # 3. Ask the Judge to score the result
        top_movie = search_hits[0]
        score, reason = get_llm_judgment(query, top_movie)
        
        results.append({
            "query": query,
            "expected": expected,
            "result": top_movie['title'],
            "score": score,
            "reason": reason
        })
        
        icon = "✅" if score >= 2 else "⚠️" if score == 1 else "❌"
        print(f"{icon} Score: {score} | {query[:40]}... -> {top_movie['title']}")

    # 4. Calculate Final Metrics
    results_df = pl.DataFrame(results)
    avg_score = results_df["score"].mean()
    success_rate = (results_df.filter(pl.col("score") >= 2).height / len(results_df)) * 100

    print(f"\n" + "="*40)
    print(f"JUDGE EVALUATION RESULTS")
    print("-" * 40)
    print(f"Average Relevance (0-3): {avg_score:.2f}")
    print(f"Semantic Accuracy (Score >= 2): {success_rate:.1f}%")
    print(f"Results saved to: {output_path}")
    print("="*40)

    results_df.write_csv(output_path)

if __name__ == "__main__":
    GT_PATH = "../data/ground_truth.csv"
    OUT_PATH = "evaluation/judge_results.csv"
    
    # Start by running on a smaller subset (e.g., 20) to check if it's working
    run_judge_evaluation(GT_PATH, OUT_PATH)