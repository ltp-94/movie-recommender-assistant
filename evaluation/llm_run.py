import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator of a movie recommendation RAG system.

Evaluate the recommendation using ONLY:
- the user query
- the expected movie
- the retrieved movie context
- the generated answer

Do not use external knowledge.

Score each criterion from 1 to 5.

1. GROUND_TRUTH_MATCH

How well does the recommendation match the expected movie?

5 = the expected movie is recommended and is an excellent match
4 = a very strong alternative to the expected movie is recommended
3 = the recommendation is reasonably relevant
2 = the recommendation is weakly related
1 = the recommendation is completely irrelevant

2. FAITHFULNESS

Are the claims in the generated answer supported by the retrieved movie context?

5 = fully supported, no hallucinations
4 = almost completely supported
3 = mostly supported
2 = several unsupported claims
1 = many unsupported or hallucinated claims

3. ANSWER_QUALITY

How useful is the generated recommendation for the user?

5 = excellent, clear and useful recommendation
4 = good recommendation with a useful explanation
3 = acceptable
2 = weak
1 = poor and does not satisfy the request

4. OVERALL

Give an overall score from 1 to 5 considering all criteria.

IMPORTANT:
The expected movie is the ground-truth reference.
However, a very strong semantically equivalent alternative can receive
a score of 4 even if it is not the exact expected movie.

Return ONLY valid JSON:

{
    "ground_truth_match": 1-5,
    "faithfulness": 1-5,
    "answer_quality": 1-5,
    "overall": 1-5,
    "reason": "short explanation"
}
"""


def judge_rag(
    user_query,
    expected_title,
    retrieved_movies,
    generated_answer
):
    """
    Evaluate the final RAG recommendation against ground truth.
    """

    # --------------------------------------------------------
    # Build retrieved movie context
    # --------------------------------------------------------

    context = ""

    for rank, movie in enumerate(retrieved_movies, start=1):

        context += f"""
Rank: {rank}
Title: {movie.get('title', '')}
Genres: {movie.get('genres', '')}
Director: {movie.get('director', '')}
Cast: {movie.get('cast', '')}
Plot: {movie.get('overview', '')}
Keywords: {movie.get('keywords', '')}
"""


    # --------------------------------------------------------
    # Judge prompt
    # --------------------------------------------------------

    user_prompt = f"""
USER QUERY:
{user_query}

EXPECTED MOVIE (GROUND TRUTH):
{expected_title}

RETRIEVED MOVIE CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}
"""


    # --------------------------------------------------------
    # Call LLM Judge
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0
    )


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    result = response.choices[0].message.content

    return json.loads(result)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "young FBI trainee looking for help of serial cannibal killer"
    )

    test_expected_title = "The Silence of the Lambs"


    test_movies = [

        {
            "title": "The Silence of the Lambs",
            "genres": "Crime, Drama, Thriller",
            "director": "Jonathan Demme",
            "cast": "Jodie Foster, Anthony Hopkins",
            "overview": (
                "A young FBI cadet must receive the help of an "
                "incarcerated and manipulative cannibal killer "
                "to help catch another serial killer."
            ),
            "keywords": (
                "FBI, serial killer, cannibal, investigation"
            )
        },

        {
            "title": "Se7en",
            "genres": "Crime, Mystery, Thriller",
            "director": "David Fincher",
            "cast": "Brad Pitt, Morgan Freeman",
            "overview": (
                "Two detectives investigate a series of murders "
                "based on the seven deadly sins."
            ),
            "keywords": (
                "serial killer, detective, murder"
            )
        }
    ]


    test_answer = """
### The Silence of the Lambs

This is an excellent match because it follows a young FBI
trainee who seeks the help of imprisoned cannibal killer
Hannibal Lecter while investigating another serial killer.

### Se7en

This is also a crime thriller involving serial murders,
although it does not specifically feature an FBI trainee
working with a cannibal killer.
"""


    # --------------------------------------------------------
    # Run Judge
    # --------------------------------------------------------

    result = judge_rag(
        user_query=test_query,
        expected_title=test_expected_title,
        retrieved_movies=test_movies,
        generated_answer=test_answer
    )


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nJudge result:")

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

