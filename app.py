import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
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

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load the dataset
        movies_data = pd.read_csv("movies.csv")
        
        # Selected features
        selected_features = ['genres', 'keywords', 'cast', 'director', 'overview']
        
        # Fill missing values
        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')
        
        # Combine features
        combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                          movies_data['cast'] + ' ' + movies_data['director'] + ' ' + \
                          movies_data['overview']
        
        # Fill any remaining NaN in combined features
        combined_features = combined_features.fillna('')
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)
        
        # Calculate similarity
        similarity = cosine_similarity(feature_vectors)
        
        return movies_data, similarity
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Get movie recommendations
def get_recommendations(movie_name, movies_data, similarity, num_recommendations=30):
    list_of_all_titles = movies_data['title'].tolist()
    
    # Find close match
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if len(find_close_match) == 0:
        return None, None
    
    close_match = find_close_match[0]
    
    # Find index of the movie
    index_of_the_movie = movies_data[movies_data['title'] == close_match].index[0]
    
    # Get similarity scores
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    
    # Sort by similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    # Get recommended movie titles (excluding the selected movie itself)
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies[1:num_recommendations+1]):
        index = movie[0]
        title = movies_data[movies_data.index == index]['title'].values[0]
        score = movie[1]
        recommended_movies.append({'rank': i+1, 'title': title, 'score': score})
    
    return close_match, recommended_movies

# Main app
def main():
    # Header
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Discover movies similar to your favorites!")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading movie database..."):
        movies_data, similarity = load_and_preprocess_data()
    
    if movies_data is None:
        st.error("❌ Failed to load movie data. Please ensure 'movies.csv' is in the same directory.")
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
        st.markdown("### 📊 Database Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", f"{len(movies_data):,}")
        with col2:
            st.metric("Features Used", "5")
        
        st.markdown("---")
        st.markdown("""
        ### 🎯 Features Used:
        - **Genres**
        - **Keywords**
        - **Cast**
        - **Director**
        - **Overview**
        
        ### 🔍 How it works:
        1. Enter or select a movie
        2. AI finds similar movies
        3. Enjoy recommendations!
        """)
    
    # Main content
    st.subheader("🔍 Find Similar Movies")
    
    # Search options
    search_option = st.radio(
        "Choose input method:",
        ["Select from dropdown", "Type movie name"],
        horizontal=True
    )
    
    if search_option == "Select from dropdown":
        movie_name = st.selectbox(
            "Select a movie:",
            options=sorted(movies_data['title'].tolist()),
            index=0
        )
    else:
        movie_name = st.text_input(
            "Enter movie name:",
            placeholder="e.g., The Dark Knight, Inception, Avatar..."
        )
    
    # Recommend button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        recommend_button = st.button("🎯 Get Recommendations", use_container_width=True)
    
    # Show recommendations
    if recommend_button and movie_name:
        with st.spinner("Finding similar movies..."):
            close_match, recommendations = get_recommendations(
                movie_name, movies_data, similarity, num_recommendations
            )
            
            if recommendations is None:
                st.error(f"❌ No close match found for '{movie_name}'. Please try another movie name.")
            else:
                # Show matched movie
                if close_match != movie_name:
                    st.info(f"🎯 Showing recommendations for: **{close_match}**")
                else:
                    st.success(f"✨ Movies similar to **{close_match}**:")
                
                st.markdown("---")
                
                # Display recommendations in a nice format
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
                
                # Download recommendations
                st.markdown("---")
                df_recommendations = pd.DataFrame(recommendations)
                csv = df_recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations (CSV)",
                    data=csv,
                    file_name=f"recommendations_{close_match.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888;">
            <p>Made with ❤️ using Streamlit & Scikit-learn | Content-Based Filtering</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()