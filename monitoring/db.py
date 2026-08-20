# monitoring/db.py
import os
import psycopg2
from datetime import datetime
from zoneinfo import ZoneInfo

TZ_INFO = os.getenv("TZ", "Europe/Berlin")
tz = ZoneInfo(TZ_INFO)

def get_db_connection():
    # If POSTGRES_HOST is not set, we default to 'localhost' for local dev
    # In docker-compose.yml, we explicitly set it to 'postgres'
    host = os.getenv("POSTGRES_HOST", "localhost") 
    
    return psycopg2.connect(
        host=host,
        database=os.getenv("POSTGRES_DB", "movie_db"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # We don't want to drop tables in production, but for dev it's okay
            cur.execute("CREATE TABLE IF NOT EXISTS conversations ("
                        "id TEXT PRIMARY KEY, "
                        "question TEXT NOT NULL, "
                        "answer TEXT NOT NULL, "
                        "model_used TEXT NOT NULL, "
                        "response_time FLOAT NOT NULL, "
                        "prompt_tokens INTEGER NOT NULL, "
                        "completion_tokens INTEGER NOT NULL, "
                        "total_tokens INTEGER NOT NULL, "
                        "total_cost FLOAT NOT NULL, "
                        "timestamp TIMESTAMP WITH TIME ZONE NOT NULL)")
            
            cur.execute("CREATE TABLE IF NOT EXISTS feedback ("
                        "id SERIAL PRIMARY KEY, "
                        "conversation_id TEXT REFERENCES conversations(id), "
                        "feedback INTEGER NOT NULL, "
                        "timestamp TIMESTAMP WITH TIME ZONE NOT NULL)")
        conn.commit()
    finally:
        conn.close()

def save_conversation(conversation_id, question, answer_data):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (id, question, answer, model_used, response_time, "
                "prompt_tokens, completion_tokens, total_tokens, total_cost, timestamp) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (conversation_id, question, answer_data["answer"], answer_data["model"],
                 answer_data["time"], answer_data["p_tokens"], answer_data["c_tokens"],
                 answer_data["t_tokens"], answer_data["cost"], datetime.now(tz))
            )
        conn.commit()
    finally:
        conn.close()

def save_feedback(conversation_id, feedback_val):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, %s)",
                (conversation_id, feedback_val, datetime.now(tz))
            )
        conn.commit()
    finally:
        conn.close()