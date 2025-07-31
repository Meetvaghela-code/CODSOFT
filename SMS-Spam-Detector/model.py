import pandas as pd
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 📥 Load dataset
data = pd.read_csv(r'data/spam.csv', encoding='latin1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 🧹 Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)         # remove links
    text = re.sub(r'<.*?>', '', text)                         # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)                      # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()                  # remove extra whitespace
    return text

data['cleaned_message'] = data['message'].apply(clean_text)

# 🎯 Features and labels
X = data['cleaned_message']
y = data['label']

# 📊 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.9, ngram_range=(1, 2), max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 🤖 Train Linear SVM model
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

# 📈 Evaluate
y_pred_svm = svm.predict(X_test_tfidf)
print("✅ SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("✅ Classification Report:\n", classification_report(y_test, y_pred_svm))

# 💾 Save the model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(svm, model_file)

# 💾 Save the vectorizer
with open('tfidf.pkl', 'wb') as vec_file:
    pickle.dump(tfidf, vec_file)

# 🔍 Optional: Define prediction function
def predict_spam(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = svm.predict(vector)[0]
    return "Spam" if prediction == 1 else "Ham"

