# AI-Based Fake News Detection

This project implements a fake news detection system using natural language processing and machine learning.

The system classifies news articles as real or fake using TF-IDF feature extraction and Logistic Regression.

## Tools Used
- Python
- Scikit-learn
- NLTK
- Flask
- Pandas

## How It Works
1. News articles are cleaned and vectorized using TF-IDF.
2. A Logistic Regression model is trained on labeled data.
3. The trained model predicts whether an article is real or fake.
4. A Flask web app provides a simple interface for testing.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Train the model:
   python train.py
3. Run the web app:
   python app.py

## Note
This project is for educational purposes only.
