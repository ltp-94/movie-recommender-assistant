import os
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import your custom logic
from rag.llm_recommendation import generate_recommendation
from rag.hybrid_search import search_rrf_pipeline
from monitoring.db import save_conversation, save_feedback, init_db

app = FastAPI(title="Movie Recommender API", description="RRF + Reranking + RAG")

# --- Pydantic Schemas ---

class QueryRequest(BaseModel):
    query: str

class MovieInfo(BaseModel):
    title: str
    year: Optional[int]
    genres: Optional[str]
    director: Optional[str]
    cast: Optional[str]
    runtime: Optional[int] = 0        # Додайте Optional та дефолт 0
    imdb_rating: Optional[float] = 0.0 # Вже добре
    vote_average: Optional[float] = 0.0 # Вже добре
    final_score: float


class RecommendationResponse(BaseModel):
    conversation_id: str
    recommendation: str
    top_movies: List[MovieInfo]
    response_time: float

class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: int 


# --- Endpoints ---
from monitoring.db import save_conversation, save_feedback, init_db # Додайте init_db

@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing Database...")
    try:
        init_db()
        print("✅ Database tables ensured.")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_movies(request: QueryRequest):
    start_time = time.time()
    user_query = request.query

    try:
        # 1. Run the RRF Search Pipeline (Rewrite -> Retrieval -> RRF -> Rerank)
        # We fetch enough candidates to ensure high quality
        final_results = search_rrf_pipeline(user_query, top_n=30)

        if not final_results:
            raise HTTPException(status_code=404, detail="No relevant movies found.")

        # 2. Generate the LLM Recommendation (RAG)
        # CRITICAL: We only pass the TOP 5 to avoid the "Request too large" (TPM) error
        ai_response = generate_recommendation(user_query)

        # 3. Calculate metrics for Monitoring
        end_time = time.time()
        duration = end_time - start_time
        conversation_id = str(uuid.uuid4())

        # 4. Save to PostgreSQL (For your Grafana Dashboard)
        # We estimate tokens if your generate_recommendation doesn't return them explicitly
        # but you can adjust these values based on your actual LLM helper output
        answer_data = {
            "answer": ai_response,
            "model": "gpt-oss-120b", # Based on your previous error log
            "time": duration,
            "p_tokens": 1500,  # Estimated
            "c_tokens": 500,   # Estimated
            "t_tokens": 2000,  # Estimated
            "cost": 0.0002     # Estimated
        }
        
        save_conversation(conversation_id, user_query, answer_data)

        # 5. Return JSON response
        return {
            "conversation_id": conversation_id,
            "recommendation": ai_response,
            "top_movies": [
                {
                    "title": m.get("title"),
                    "genres": m.get("genres"),
                    "director": m.get("director"),
                    "cast": m.get("cast"),
                    "year": m.get("release_year"),
                    "runtime": m.get("runtime"),
                    "imdb_rating": m.get("imdb_rating", 0.0),
                    "vote_average": m.get("vote_average", 0.0),
                    "final_score": round(m.get("final_score", 0), 2)
                } for m in final_results
            ],
            "response_time": round(duration, 2)
        }

    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/feedback")
async def post_feedback(request: FeedbackRequest):
    try:
        # Перевіряємо, чи feedback є коректним значенням (опціонально)
        if request.feedback not in [1, -1]:
            raise HTTPException(status_code=400, detail="Feedback must be 1 or -1")

        # Зберігаємо в базу даних
        save_feedback(request.conversation_id, request.feedback)
        
        return {"message": f"Feedback received for conversation {request.conversation_id}"}
    
    except Exception as e:
        # Якщо conversation_id не існує в базі, Postgres видасть помилку Foreign Key
        print(f"❌ Feedback Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not save feedback. Make sure conversation_id exists.")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_device": "cpu"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)