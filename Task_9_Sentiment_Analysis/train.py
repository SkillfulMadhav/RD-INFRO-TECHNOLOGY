import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("sentiment.csv")

data = data.dropna(subset=["clean_text", "category"])
data["clean_text"] = data["clean_text"].astype(str)

X = data["clean_text"]
y = data["category"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully")
