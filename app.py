import streamlit as st
from movie_recommendation import combined_recommend, movies

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie and get personalized movie recommendations "
    "using a hybrid recommendation system."
)

st.divider()

# Movie selection
st.subheader("🎥 Select a Movie")

movie_title = st.selectbox(
    "Choose your favorite movie:",
    movies["title"].tolist()
)

# Number of recommendations
num_recommendations = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)

# Recommendation button
if st.button("✨ Recommend Movies", use_container_width=True):

    recommendations = combined_recommend(
        movie_title,
        num_recommendations
    )

    st.divider()

    st.subheader(f"Recommended Movies for: {movie_title}")

    for i, movie in enumerate(recommendations, 1):
        st.write(f"**{i}.** 🎬 {movie}")

st.divider()

st.caption(
    "Movie Recommendation System | MovieLens 1M Dataset | "
    "Content-Based + Collaborative Filtering"
)