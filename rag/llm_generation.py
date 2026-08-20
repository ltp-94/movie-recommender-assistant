import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"), 
    base_url="https://api.groq.com/openai/v1"
)


def generate_recommendation(user_query: str, search_results: list):
    """
    Takes the top search results and generates a conversational AI response.
    """
    if not search_results:
        return "I'm sorry, I couldn't find any movies matching your request in my database."

    # 1. Prepare context for the LLM
    context = ""
    for movie in search_results:
        context += f"Title: {movie['title']}\n"
        context += f"Genres: {movie['genres']}\n"
        context += f"Director: {movie['director']}\n"
        context += f"Writers: {movie['writers']}\n"
        context += f"Cast: {movie['cast']}\n"
        context += f"Plot: {movie['overview']}\n"
        context += f"Keywords: {movie['keywords']}\n"
        context += f"Poster URL: {movie.get('poster_url', 'N/A')}\n"
        context += f"Movie Link: {movie.get('movie_link', 'N/A')}\n"


    system_prompt = """
        You are a professional movie critic. 
        IMPORTANT: You may ONLY recommend movies that are provided in the "Movie Context" below.
        If a movie is not in the context, DO NOT mention it.
        Explain WHY you are recommending these specific movies based ONLY on the provided plot and cast.
        Use a friendly, conversational tone and bullet points for multiple recommendations.

        For every movie you recommend:
        1. Explain WHY you are recommending it based on the plot.
        2. Format the title as a clickable Markdown link using the 'Movie Link' provided.
        3. Display the poster image using Markdown syntax: ![poster](Poster URL)
        
        Example format:
        ### [Movie Title](Movie Link)
        ![poster](Poster URL)
        **Why this movie:** [Your explanation...]
    """

    user_prompt = f"User Request: {user_query}\n\nMovie Context:\n{context}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# import os
# from google import genai
# from dotenv import load_dotenv

# load_dotenv()
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# def generate_recommendation(user_query: str, search_results: list):
#     if not search_results:
#         return "I'm sorry, I couldn't find any movies matching your request."

#     context = ""
#     # Only take top 8 to stay safe with token limits and keep response fast
#     for movie in search_results[:8]:
#         context += f"Title: {movie['title']}\n"
#         context += f"Plot: {movie.get('overview', 'N/A')}\n"
#         context += f"Poster URL: {movie.get('poster_url', 'N/A')}\n"
#         context += f"Movie Link: {movie.get('movie_link', 'N/A')}\n"
#         context += "------------------\n"

#     prompt = f"""
#     You are a professional movie critic. 
#     IMPORTANT: You may ONLY recommend movies provided in the "Movie Context" below.
    
#     USER REQUEST: {user_query}
#     MOVIE CONTEXT:
#     {context}

#     INSTRUCTIONS:
#     1. Explain WHY you are recommending these movies based on the plot.
#     2. Format the title as a clickable Markdown link: [Title](Movie Link)
#     3. Display the poster image using Markdown: ![poster](Poster URL)
#     """

#     try:
#         response = client.models.generate_content(
#             model="gemini-1.5-flash",
#             contents=prompt
#         )
#         return response.text
#     except Exception as e:
#         return f"Error generating recommendation: {str(e)}"