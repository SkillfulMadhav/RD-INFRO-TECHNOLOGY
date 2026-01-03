import streamlit as st
import pickle

st.set_page_config(page_title="Sentiment Analysis")

st.title("Sentiment Analysis Test")
st.write("If you can see this, Streamlit is rendering correctly.")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

text = st.text_area("Enter text")

if st.button("Analyze"):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    st.write("Prediction:", pred)
