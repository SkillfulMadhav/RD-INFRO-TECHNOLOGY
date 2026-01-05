import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

def get_response(message):
    tokens = nltk.word_tokenize(message.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    sentence = " ".join(tokens)

    X = vectorizer.transform([sentence])
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."
