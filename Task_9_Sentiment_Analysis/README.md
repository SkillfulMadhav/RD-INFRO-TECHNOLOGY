# AI-Based Sentiment Analysis for Social Media

## Description
This project implements a machine learning–based sentiment analysis system that classifies text into sentiment categories (such as positive, negative, or neutral).  
It is built using classical NLP techniques and deployed with Streamlit for an interactive web interface.

This project is for educational purposes and demonstrates an end-to-end NLP pipeline:
data preprocessing → model training → prediction → UI.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

## Project Structure
# AI-Based Sentiment Analysis for Social Media

## Description
This project implements a machine learning–based sentiment analysis system that classifies text into sentiment categories (such as positive, negative, or neutral).  
It is built using classical NLP techniques and deployed with Streamlit for an interactive web interface.

This project is for educational purposes and demonstrates an end-to-end NLP pipeline:
data preprocessing → model training → prediction → UI.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit

## Project Structure
Task_9_Sentiment_Analysis/
├── app.py # Streamlit web app
├── train.py # Model training script
├── sentiment.csv # Dataset
├── model.pkl # Trained sentiment model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt
└── README.md


## How It Works
1. Text data is cleaned and preprocessed.
2. TF-IDF is used to convert text into numerical features.
3. A Logistic Regression model is trained on labeled sentiment data.
4. The trained model is used to predict sentiment for new user input.
5. Streamlit provides a simple UI to test predictions in real time.

## How to Run

### 1. Install dependencies


pip install -r requirements.txt


### 2. Train the model


python train.py


### 3. Run the web app


streamlit run app.py


## Notes
- Dataset rows with missing text are automatically removed.
- This is a demo project and not production-ready.
