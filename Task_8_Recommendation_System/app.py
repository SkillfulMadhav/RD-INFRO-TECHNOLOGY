import streamlit as st
from recommender import recommend
import pandas as pd

st.title("AI-Based Movie Recommendation System")

movies = pd.read_csv("movies.csv")
movie_list = movies["title"].tolist()

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
