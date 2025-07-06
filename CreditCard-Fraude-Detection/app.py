from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

# Load column names used during training
X_columns = pd.read_csv('model/columns.csv').columns.tolist()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Collect user inputs from the form
        amt = float(request.form['amt'])
        gender = request.form['gender']
        category = request.form['category']
        merchant = request.form['merchant']
        hour = int(request.form['hour'])
        day = int(request.form['day'])

        # Create a base input DataFrame
        input_data = pd.DataFrame([{
            'amt': amt,
            'gender': gender,
            'category': category,
            'merchant': merchant,
            'hour': hour,
            'day': day
        }])

        # Apply one-hot encoding (just like training)
        input_encoded = pd.get_dummies(input_data, columns=['gender', 'category', 'merchant'], drop_first=True)

        # Add missing columns and sort to match training columns
        for col in X_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X_columns]  # Ensure correct column order

        # Scale numeric input
        scaled_input = scaler.transform(input_encoded)

        # Predict
        pred = model.predict(scaled_input)[0]
        prediction = "Fraudulent Transaction ⚠️" if pred == 1 else "Legitimate Transaction ✅"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
