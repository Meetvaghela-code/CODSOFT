from flask import Flask, render_template, request
import pickle
import re

# Load trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

# 🧼 Text cleaning function (match training!)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = ""
    if request.method == 'POST':
        message = request.form['message']
        if message.strip() == "":
            result = "Please enter a message."
        else:
            cleaned = clean_text(message)
            vector = tfidf.transform([cleaned])
            prediction = model.predict(vector)[0]
            result = "Spam" if prediction == 1 else "Ham"
    return render_template('index.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
