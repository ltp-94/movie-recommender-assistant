import uuid
import time
from rag.search_popularity import search_movies
from rag.llm_generation import generate_recommendation
from monitoring.db import save_conversation, save_feedback

# ... UI layout (st.title, st.text_input) ...

if st.button("Search"):
    # 1. Create a unique ID for this search session
    st.session_state.conv_id = str(uuid.uuid4())
    
    start_time = time.time()
    # 2. Get Results
    results = search_movies(query)
    # 3. Get AI Response
    answer, usage = generate_recommendation(query, results)
    end_time = time.time()
    
    # 4. Save to Database
    answer_data = {
        "answer": answer,
        "model": "llama-3.3-70b-versatile",
        "time": end_time - start_time,
        "p_tokens": usage.prompt_tokens,
        "c_tokens": usage.completion_tokens,
        "t_tokens": usage.total_tokens,
        "cost": (usage.total_tokens / 1_000_000) * 0.60
    }
    save_conversation(st.session_state.conv_id, query, answer_data)
    
    # 5. Display Answer
    st.write(answer)

# ... Under the answer, add feedback buttons ...
if "conv_id" in st.session_state:
    col1, col2 = st.columns(10)
    with col1:
        if st.button("👍"):
            save_feedback(st.session_state.conv_id, 1)
            st.toast("Feedback saved!")
    with col2:
        if st.button("👎"):
            save_feedback(st.session_state.conv_id, -1)
            st.toast("Feedback saved!")