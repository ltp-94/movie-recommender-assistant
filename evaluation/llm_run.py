import os
import json
import polars as pl
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
# Імпортуємо вашу функцію RRF пошуку
from chatgpt.test import search_rrf_evaluation_pipeline as search_movies

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Для оцінки списків краще використовувати llama-3.1-70b, 
# вона краще розуміє контекст між декількома фільмами.
JUDGE_MODEL = "openai/gpt-oss-120b" 

JUDGE_PROMPT = """
You are a search quality auditor. Your job is to evaluate if a movie search engine is providing relevant recommendations.

USER QUERY: {query}
EXPECTED MOVIE (Ground Truth): {expected}

RETRIEVED LIST:
{results_formatted}

INSTRUCTIONS:
1. Assign a score from 0 to 3 for EACH retrieved movie.
   - 3: Perfect match (either the Expected Movie or an identical plot/intent).
   - 2: Highly relevant (same genre, similar plot, same director/vibe).
   - 1: Partially relevant (same actor but wrong genre, or weak connection).
   - 0: Irrelevant (no connection).
2. Provide a brief reasoning for each score.
3. Be objective. If the search found a great alternative that isn't the Expected Movie, score it a 2.

RETURN JSON ONLY:
{{
  "evaluations": [
    {{"rank": 1, "title": "...", "score": 3, "reason": "..."}},
    ...
  ]
}}
"""

def judge_search_results(query, expected, hits):
    formatted_hits = ""
    for i, m in enumerate(hits, 1):
        formatted_hits += f"Rank {i}: {m['title']} | Overview: {m['overview'][:200]}...\n"
    
    prompt = JUDGE_PROMPT.format(
        query=query, 
        expected=expected, 
        results_formatted=formatted_hits
    )
    
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content).get("evaluations", [])
    except Exception as e:
        print(f"Judge error: {e}")
        return []

def run_evaluation(gt_path, output_path, top_n=5):
    df_gt = pl.read_csv(gt_path)
    all_judgments = []

    print(f"Evaluating {len(df_gt)} queries (Top {top_n})...")

    for row in tqdm(df_gt.iter_rows(named=True)):
        query = row['query']
        expected = row['expected_title']
        
        # Виклик вашого пошуку
        hits = search_movies(query, top_n=top_n)
        
        if not hits:
            continue
            
        # Оцінка списку через LLM
        evals = judge_search_results(query, expected, hits)
        
        for e in evals:
            all_judgments.append({
                "query": query,
                "expected": expected,
                "retrieved": e['title'],
                "rank": e['rank'],
                "llm_score": e['score'],
                "llm_reasoning": e['reason']
            })

    # Збереження та аналіз
    res_df = pl.DataFrame(all_judgments)
    
    # Розрахунок середньої релевантності всього топу
    avg_relevance = res_df['llm_score'].mean()
    
    # Розрахунок "Semantic Precision": відсоток результатів зі скором 2 або 3
    semantic_precision = (res_df.filter(pl.col("llm_score") >= 2).height / len(res_df)) * 100

    print("\n" + "="*40)
    print("LLM JUDGE METRICS")
    print("-" * 40)
    print(f"Average Relevance Score (0-3): {avg_relevance:.2f}")
    print(f"Semantic Precision @ {top_n}: {semantic_precision:.1f}%")
    print(f"Results saved to: {output_path}")
    print("="*40)
    
    res_df.write_csv(output_path)

if __name__ == "__main__":
    run_evaluation("../data/ground_truth.csv", "evaluation/llm_judge_full_results.csv")