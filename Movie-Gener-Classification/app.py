from flask import Flask, render_template, request
import pickle
import re, string

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        description = request.form.get("description", "")
        if description:
            desc_clean = preprocess_text(description)
            vec = vectorizer.transform([desc_clean])
            prediction = model.predict(vec)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
