import json
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        sentences.append(" ".join(tokens))
        labels.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X, labels)

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model trained successfully")
