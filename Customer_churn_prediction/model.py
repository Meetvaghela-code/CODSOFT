import pandas as pd
import pickle
data = pd.read_csv(r'data/Churn_Modelling.csv')
print(data.head())

# Check basic info
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Descriptive statistics
print(data.describe())

# Check column names
print(data.columns)

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# One-hot encoding Geography, and Label Encoding Gender
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Male = 1, Female = 0

X = data.drop('Exited', axis=1)
y = data['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)