import json
import pandas as pd

from chatgpt.test import search_rrf_evaluation_pipeline
from rag.llm_generation import generate_recommendation
from llm_run import judge_rag


GROUND_TRUTH_PATH = "/workspaces/movie-recommender-assistant/data/ground_truth.csv"


def evaluate():

    df = pd.read_csv(GROUND_TRUTH_PATH)

    all_results = []

    for i, row in df.iterrows():

        query = row["query"]
        expected_title = row["expected_title"]

        print("\n" + "=" * 70)
        print(f"Query {i + 1}/{len(df)}")
        print("=" * 70)

        print(f"Query:    {query}")
        print(f"Expected: {expected_title}")

        # --------------------------------------------------
        # 1. Run your RAG retrieval
        # --------------------------------------------------

        retrieved_movies = search_rrf_evaluation_pipeline(
            query,
            top_n=10
        )

        print("\nRetrieved movies:")

        for rank, movie in enumerate(
            retrieved_movies,
            start=1
        ):
            print(
                f"{rank}. {movie['title']}"
            )

        # --------------------------------------------------
        # 2. Generate recommendation
        # --------------------------------------------------

        answer = generate_recommendation(
            query,
            retrieved_movies
        )

        print("\nGenerated answer:")
        print(answer)

        # --------------------------------------------------
        # 3. LLM Judge
        # --------------------------------------------------

        judge_result = judge_rag(
            user_query=query,
            expected_title=expected_title,
            retrieved_movies=retrieved_movies,
            generated_answer=answer
        )

        print("\nJudge:")
        print(
            json.dumps(
                judge_result,
                indent=4,
                ensure_ascii=False
            )
        )

        # --------------------------------------------------
        # 4. Save result
        # --------------------------------------------------

        all_results.append({
            "query": query,
            "expected_title": expected_title,
            "retrieved_titles": [
                movie["title"]
                for movie in retrieved_movies
            ],
            "generated_answer": answer,
            "ground_truth_match":
                judge_result["ground_truth_match"],
            "faithfulness":
                judge_result["faithfulness"],
            "answer_quality":
                judge_result["answer_quality"],
            "overall":
                judge_result["overall"],
            "reason":
                judge_result["reason"]
        })

    return all_results


if __name__ == "__main__":

    results = evaluate()

    # Save detailed results
    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ------------------------------------------------------
    # Calculate averages
    # ------------------------------------------------------

    avg_ground_truth = sum(
        r["ground_truth_match"]
        for r in results
    ) / len(results)

    avg_faithfulness = sum(
        r["faithfulness"]
        for r in results
    ) / len(results)

    avg_answer_quality = sum(
        r["answer_quality"]
        for r in results
    ) / len(results)

    avg_overall = sum(
        r["overall"]
        for r in results
    ) / len(results)

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Ground Truth Match: {avg_ground_truth:.2f}/5"
    )

    print(
        f"Faithfulness:       {avg_faithfulness:.2f}/5"
    )

    print(
        f"Answer Quality:     {avg_answer_quality:.2f}/5"
    )

    print(
        f"Overall:            {avg_overall:.2f}/5"
    )

    print("\nDetailed results saved to:")
    print("evaluation_results.json")