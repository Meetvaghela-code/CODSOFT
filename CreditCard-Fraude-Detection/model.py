import os
import kagglehub
import pandas as pd
import pickle

# 📁 Ensure model directory exists
os.makedirs("model", exist_ok=True)

# 📥 Download dataset
path = kagglehub.dataset_download("kartik2112/fraud-detection")
print("Path to dataset files:", path)
print("Content:", os.listdir(path))

# 📄 Load data
train = pd.read_csv(os.path.join(path, "fraudTrain.csv"))

# 🧠 Feature engineering
train['trans_date_trans_time'] = pd.to_datetime(train['trans_date_trans_time'])
train['hour'] = train['trans_date_trans_time'].dt.hour
train['day'] = train['trans_date_trans_time'].dt.day

# 🧹 Drop irrelevant columns
cols_to_drop = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'street',
                'city', 'state', 'zip', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'job']
train = train.drop(columns=cols_to_drop)

# 🔠 One-hot encode categorical variables
categorical_cols = ['category', 'gender', 'merchant']
train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)

# 🎯 Prepare features and target
X = train.drop('is_fraud', axis=1)
y = train['is_fraud']

# 🧪 Keep only numeric features
X = X.select_dtypes(include=[int, float, bool])

# 💾 Save column names for web UI prediction
pd.DataFrame(columns=X.columns).to_csv("model/columns.csv", index=False)

# 🔀 Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚖️ Undersample non-fraud class to balance
fraud = X_train[y_train == 1]
not_fraud = X_train[y_train == 0].sample(n=len(fraud), random_state=42)

X_balanced = pd.concat([fraud, not_fraud])
y_balanced = pd.Series([1]*len(fraud) + [0]*len(not_fraud))

print("\nClass balance after undersampling:\n", y_balanced.value_counts())

# 🧮 Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)
X_val_scaled = scaler.transform(X_val)

# 🌲 Train RandomForest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
clf.fit(X_scaled, y_balanced)

# 🧪 Evaluate model
y_pred = clf.predict(X_val_scaled)

print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Accuracy Score:", accuracy_score(y_val, y_pred))

# 💾 Save model, scaler, and columns
with open('model/model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model, scaler, and column names saved to /model successfully.")
