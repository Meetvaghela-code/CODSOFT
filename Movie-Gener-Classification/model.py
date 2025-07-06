import os
import re
import string
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # More powerful than Naive Bayes
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Download dataset
path = kagglehub.dataset_download("hijest/genre-classification-dataset-imdb")
subfolder = os.path.join(path, "Genre Classification Dataset")
train_file = os.path.join(subfolder, "train_data.txt")

# Load data
genres = []
descriptions = []

with open(train_file, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ::: ")
        if len(parts) >= 4:
            genres.append(parts[2].strip())
            descriptions.append(parts[3].strip())

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

# Create DataFrame
df = pd.DataFrame({"Genre": genres, "Description": descriptions})
df.dropna(inplace=True)
df = df[df["Description"].str.len() > 30]  # remove very short ones
df["Description"] = df["Description"].apply(preprocess_text)

# Train/Test Split
X = df["Description"]
y = df["Genre"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Model: Logistic Regression with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_val_tfidf)
print("\n✅ Accuracy:", accuracy_score(y_val, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_val, y_pred))

# Genre prediction function
def predict_genre(description):
    description = preprocess_text(description)
    vec = vectorizer.transform([description])
    return model.predict(vec)[0]


# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
