# import os
# import polars as pl
# from openai import OpenAI
# from dotenv import load_dotenv

# from chatgpt.test import search_rrf_evaluation_pipeline as search_movies


# load_dotenv()


# client = OpenAI(
#     api_key=os.getenv("GROQ_API_KEY"),
#     base_url="https://api.groq.com/openai/v1"
# )


# JUDGE_MODEL = "openai/gpt-oss-20b"


# JUDGE_PROMPT = """
# You are an expert movie search quality evaluator.

# USER QUERY:
# {query}

# EXPECTED MOVIE:
# {expected}

# RETRIEVED MOVIE:
# {title}

# MOVIE OVERVIEW:
# {overview}

# Evaluate how good the retrieved movie is for the user's query.

# SCORING:

# 3 = EXCELLENT
# The retrieved movie is the expected movie or an almost exact match.

# 2 = GOOD
# The movie is highly relevant to the query but is not the expected movie.

# 1 = PARTIAL
# The movie has some connection to the query but misses the main intent.

# 0 = IRRELEVANT
# The movie does not match the user's request.

# Return exactly:

# Score: [0-3]
# Reason: [one short sentence]
# """


# def get_llm_judgment(query, expected, movie):

#     prompt = JUDGE_PROMPT.format(
#         query=query,
#         expected=expected,
#         title=movie.get("title", ""),
#         overview=movie.get("overview", "")
#     )

#     try:

#         response = client.chat.completions.create(
#             model=JUDGE_MODEL,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0
#         )

#         content = response.choices[0].message.content
#         print("\nJUDGE RAW RESPONSE:")
#         print(content)

#         score_line = next(
#             line for line in content.splitlines()
#             if "Score:" in line
#         )

#         reason_line = next(
#             line for line in content.splitlines()
#             if "Reason:" in line
#         )

#         score = int(
#             score_line.split(":", 1)[1].strip()
#         )

#         reason = reason_line.split(
#             ":", 1
#         )[1].strip()

#         return score, reason

#     except Exception as e:
#         print("\nJUDGE ERROR:")
#         print(type(e).__name__)
#         print(e)

#         return 0, f"Error: {e}"


# def run_judge_evaluation(
#     gt_path,
#     output_path,
#     limit=None
# ):

#     # Load ground truth
#     df = pl.read_csv(gt_path)

#     if limit:
#         df = df.head(limit)

#     results = []

#     print(
#         f"Starting evaluation on {len(df)} queries...\n"
#     )

#     for row in df.iter_rows(named=True):

#         query = row["query"]
#         expected = row["expected_title"]

#         # -----------------------------------------
#         # RAG SEARCH
#         # -----------------------------------------

#         search_hits = search_movies(
#             query,
#             top_n=1
#         )

#         if not search_hits:

#             results.append({
#                 "query": query,
#                 "expected": expected,
#                 "result": "N/A",
#                 "score": 0,
#                 "reason": "No results found"
#             })

#             continue

#         # -----------------------------------------
#         # TOP RESULT
#         # -----------------------------------------

#         movie = search_hits[0]

#         score, reason = get_llm_judgment(
#             query,
#             expected,
#             movie
#         )

#         results.append({
#             "query": query,
#             "expected": expected,
#             "result": movie["title"],
#             "score": score,
#             "reason": reason
#         })

#         print(
#             f"{score}/3 | "
#             f"{expected} -> {movie['title']}"
#         )

#     # -----------------------------------------
#     # METRICS
#     # -----------------------------------------

#     results_df = pl.DataFrame(results)

#     avg_score = results_df["score"].mean()

#     success_rate = (
#         results_df
#         .filter(pl.col("score") >= 2)
#         .height
#         / len(results_df)
#         * 100
#     )

#     print("\n" + "=" * 50)
#     print("LLM JUDGE RESULTS")
#     print("=" * 50)

#     print(
#         f"Average Relevance: {avg_score:.2f}/3"
#     )

#     print(
#         f"Success Rate: {success_rate:.1f}%"
#     )

#     print(
#         f"Results saved to: {output_path}"
#     )

#     print("=" * 50)

#     results_df.write_csv(output_path)


# if __name__ == "__main__":

#     GT_PATH = "../data/ground_truth.csv"

#     OUT_PATH = "judge_results.csv"

#     run_judge_evaluation(
#         GT_PATH,
#         OUT_PATH,
#         limit=20
#     )

import os
import json
import polars as pl
from openai import OpenAI
from dotenv import load_dotenv
from chatgpt.test import search_rrf_evaluation_pipeline as search_movies

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Краще використовувати Llama-3.1-8b або 70b на Groq
JUDGE_MODEL = "llama-3.1-8b-instant" 

JUDGE_PROMPT = """
You are an expert movie search evaluator.
Compare the USER QUERY and EXPECTED MOVIE against the RETRIEVED MOVIE.

QUERY: {query}
EXPECTED: {expected}
RETRIEVED: {title}
OVERVIEW: {overview}

SCORING:
3: EXCELLENT - Exact match or identical intent.
2: GOOD - Highly relevant, fits query perfectly but different title.
1: PARTIAL - Related (same actor/genre) but misses core intent.
0: IRRELEVANT - No connection.

Return ONLY a JSON object:
{{"score": int, "reason": "string"}}
"""

def get_llm_judgment(query, expected, movie):
    prompt = JUDGE_PROMPT.format(
        query=query,
        expected=expected,
        title=movie.get("title", "N/A"),
        overview=movie.get("overview", "N/A")
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, # Вмикаємо JSON mode
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)
        return result.get("score", 0), result.get("reason", "N/A")

    except Exception as e:
        return 0, f"Error: {str(e)}"

def run_judge_evaluation(gt_path, output_path, limit=None):
    df = pl.read_csv(gt_path)
    if limit: df = df.head(limit)

    results = []
    print(f"🚀 Starting evaluation on {len(df)} queries...\n")

    for row in df.iter_rows(named=True):
        query = row["query"]
        expected = row["expected_title"]

        search_hits = search_movies(query, top_n=1)
        
        if not search_hits:
            results.append({
                "query": query, "expected": expected, "result": "N/A",
                "score": 0, "reason": "No results found"
            })
            continue

        movie = search_hits[0]
        score, reason = get_llm_judgment(query, expected, movie)

        results.append({
            "query": query,
            "expected": expected,
            "result": movie["title"],
            "score": score,
            "reason": reason
        })

        icon = "✅" if score >= 2 else "❌"
        print(f"{icon} {score}/3 | {expected} -> {movie['title']}")

    # Метрики
    results_df = pl.DataFrame(results)
    print("\n" + "="*50)
    print(f"AVG RELEVANCE: {results_df['score'].mean():.2f}/3")
    print(f"SUCCESS RATE: {(results_df.filter(pl.col('score') >= 2).height / len(results_df)):.1%}")
    print("="*50)
    
    results_df.write_csv(output_path)

if __name__ == "__main__":
    run_judge_evaluation("../data/ground_truth.csv", "judge_results.csv", limit=20)