from flask import Flask, request, render_template_string
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

HTML = """
<h2>Fake News Detection</h2>
<form method="post">
<textarea name="news" rows="10" cols="80"></textarea><br><br>
<input type="submit">
</form>
{% if result %}
<h3>{{ result }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        news = request.form["news"]
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]
        result = "Real News" if prediction == 1 else "Fake News"
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
