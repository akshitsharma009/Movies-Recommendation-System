import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 10px;
        font-size: 16px;
    }
    .movie-card {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load and preprocess data
# --------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    try:
        movies_data = pd.read_csv("movies.csv")

        selected_features = ['genres', 'keywords', 'cast', 'director', 'overview']

        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')

        combined_features = (
            movies_data['genres'] + ' ' +
            movies_data['keywords'] + ' ' +
            movies_data['cast'] + ' ' +
            movies_data['director'] + ' ' +
            movies_data['overview']
        )

        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)

        similarity = cosine_similarity(feature_vectors)

        return movies_data, similarity

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --------------------------------------------------
# Recommendation logic
# --------------------------------------------------
def get_recommendations(movie_name, movies_data, similarity, num_recommendations=30):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return None, None

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]

    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:num_recommendations + 1]):
        index = movie[0]
        title = movies_data.iloc[index]['title']
        score = movie[1]
        recommended_movies.append({
            'rank': i + 1,
            'title': title,
            'score': score
        })

    return close_match, recommended_movies

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Discover movies similar to your favorites!")
    st.markdown("---")

    with st.spinner("Loading movie database..."):
        movies_data, similarity = load_and_preprocess_data()

    if movies_data is None:
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )

        st.markdown("---")
        st.metric("Total Movies", f"{len(movies_data):,}")
        st.metric("Features Used", "5")

        st.markdown("""
        ### 🎯 Features Used
        - Genres
        - Keywords
        - Cast
        - Director
        - Overview
        """)

    # Main section
    st.subheader("🔍 Find Similar Movies")

    search_option = st.radio(
        "Choose input method:",
        ["Select from dropdown", "Type movie name"],
        horizontal=True
    )

    if search_option == "Select from dropdown":
        movie_name = st.selectbox(
            "Select a movie:",
            sorted(movies_data['title'].tolist())
        )
    else:
        movie_name = st.text_input(
            "Enter movie name:",
            placeholder="e.g., Inception, Avatar, Titanic..."
        )

    if st.button("🎯 Get Recommendations"):
        close_match, recommendations = get_recommendations(
            movie_name, movies_data, similarity, num_recommendations
        )

        if recommendations is None:
            st.error("❌ No close match found.")
        else:
            st.success(f"✨ Movies similar to **{close_match}**")
            st.markdown("---")

            for i in range(0, len(recommendations), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        movie = recommendations[i + j]
                        with col:
                            st.markdown(f"""
                            <div class="movie-card">
                                <h3>#{movie['rank']}</h3>
                                <h4>{movie['title']}</h4>
                                <p>Match Score: {movie['score']:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Footer / Ownership
    # --------------------------------------------------
    st.markdown("---")
    st.markdown(
        "👨‍💻 **Created by Akshit Sharma** | "
        "[GitHub](https://github.com/akshitsharma009)",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
