import streamlit as st
from rag.search_popularity import search_movies
from rag.llm_generation import generate_recommendation

# Streamlit App
st.title("🎬 Movie Recommender System")

# User input
query = st.text_input("Enter your movie query:")

top_k = st.slider("Number of results to retrieve", 5, 30, 10)

if st.button("Search"):
    with st.spinner("Searching movies..."):
        results = search_movies(query, top_n=top_k)

    if not results:
        st.warning("No movies found for your query.")
    else:
        st.subheader("AI Recommendation")
        recommendation = generate_recommendation(query, results)
        st.write(recommendation)
        st.subheader("Top Results")
        for i, m in enumerate(results, 1):
            st.write(f"**{i}. {m['title']}** ({m.get('release_year','N/A')})")
            st.write(f"Genres: {m.get('genres','N/A')}")
            st.write(f"Director: {m.get('director','N/A')}")
            st.write(f"Plot: {m.get('overview','N/A')}")
            st.write(f"Movie Link: {m.get('movie_link','N/A')}")
            st.write(f"Poster: {m.get('poster_url','N/A')}")
            st.write("---")


