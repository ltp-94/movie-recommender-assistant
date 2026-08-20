# test_system.py
import uuid
import time
from rag.search_popularity import search_movies
from rag.llm_generation import generate_recommendation
from monitoring.db import save_conversation, get_recent_conversations

def test_full_flow():
    print("🚀 Starting System Test...")
    
    query = "A high-stakes heist movie set in space"
    conv_id = str(uuid.uuid4())
    
    # 1. Test Search
    print("🔍 Testing Elasticsearch Search...")
    results = search_movies(query, top_n=5)
    print(f"✅ Found {len(results)} movies.")

    # 2. Test LLM
    print("🧠 Testing LLM Generation...")
    # NOTE: Ensure your generate_recommendation returns (answer, usage)
    answer, usage = generate_recommendation(query, results)
    print(f"✅ AI Answer received ({usage.total_tokens} tokens used).")

    # 3. Test Postgres Logging
    print("📝 Testing Postgres Logging...")
    answer_data = {
        "answer": answer,
        "model": "llama-3.3-70b-versatile",
        "time": 2.5, # dummy time
        "p_tokens": usage.prompt_tokens,
        "c_tokens": usage.completion_tokens,
        "t_tokens": usage.total_tokens,
        "cost": 0.0005
    }
    save_conversation(conv_id, query, answer_data)
    print("✅ Conversation saved to database.")

    # 4. Verify DB Content
    print("\n📊 Verifying Database content...")
    recent = get_recent_conversations(limit=1)
    if recent and recent[0]['id'] == conv_id:
        print("🎉 SUCCESS: Everything is integrated and working!")
        print(f"Database Record: {recent[0]['question']} -> {recent[0]['answer'][:50]}...")
    else:
        print("❌ ERROR: Data was not found in Postgres.")

if __name__ == "__main__":
    test_full_flow()