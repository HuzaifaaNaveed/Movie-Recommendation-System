import streamlit as st
from MRS import newdf, sim,recommend_content, hybrid_recommendation
from PIL import Image
import requests

# Title and Page Configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎥  MOVIE RECOMMENDER SYSTEM")
st.write("""
Welcome to the Movie Recommendation System! Choose one of the following options:
- **Content-Based Recommendations**: Get recommendations based on movie features like genres, cast, etc.
- **Hybrid Recommendations**: Combines content similarity and collaborative filtering for better results.
""")

# Sidebar for Recommendation Options
st.sidebar.title("Recommendation Options")
recommendation_type = st.sidebar.radio(
    "Choose Recommendation Type:",
    options=["Content-Based", "Hybrid"]
)

# User Inputs
st.markdown("---")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .stApp {
        background-image: url('https://www.transparenttextures.com/patterns/clean-gray-paper.png');
        background-size: cover;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .subtitle {
        font-size: 18px;
        font-style: italic;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.subheader("Enter Details Below")
movie = st.text_input("Enter a movie title:", placeholder="e.g., Avatar")
n_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=20, value=10)


if st.button("Get Recommendations"):
    if movie:
        try:
            if recommendation_type == "Content-Based":
                recommendations = recommend_content(movie, sim, newdf, n_recommendations)
            elif recommendation_type == "Hybrid":
                recommendations = hybrid_recommendation(movie, n_recommendations)

            if recommendations:
                st.success(f"Top {n_recommendations} Recommended Movies for '{movie}':")
                for idx, rec in enumerate(recommendations, start=1):
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.write("🎥")
                    with col2:
                        st.write(f"** {rec}**")
            else:
                st.warning(f"No recommendations found for '{movie}'. Try a different title.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a movie title!")

# Footer
st.markdown("---")
st.markdown('<p class="footer">MOVIE RECOMMENDER SYSTEM &copy; 2024</p>', unsafe_allow_html=True)
