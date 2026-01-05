# AI-Based Customer Support Chatbot

This project is an AI-powered customer support chatbot built using Natural Language Processing techniques. It is designed to handle common customer queries in real time by classifying user intent and returning appropriate responses.

The chatbot is implemented with a modular architecture and exposed through a Flask-based REST API.

## Features
- Intent-based query classification
- NLP preprocessing using NLTK
- Machine learning model for intent prediction
- REST API for chatbot interaction
- Easily extensible intent dataset

## Tech Stack
- Python
- NLTK
- Scikit-learn
- Flask
- NumPy

## Project Structure
AI_Chatbot/
├── app.py
├── chatbot.py
├── train.py
├── intents.json
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md


## How It Works
1. User input is preprocessed using tokenization and lemmatization.
2. Text is converted into numerical features using CountVectorizer.
3. A trained Logistic Regression model predicts the intent.
4. A response corresponding to the predicted intent is returned.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
