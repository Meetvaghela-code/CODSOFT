from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    geography = request.form['Geography']

    # One-hot encoding manually
    geo_germany = 1 if geography == 'Germany' else 0
    geo_spain = 1 if geography == 'Spain' else 0

    features = [
        float(request.form['CreditScore']),
        int(request.form['Gender']),
        int(request.form['Age']),
        int(request.form['Tenure']),
        float(request.form['Balance']),
        int(request.form['NumOfProducts']),
        int(request.form['HasCrCard']),
        int(request.form['IsActiveMember']),
        float(request.form['EstimatedSalary']),
        geo_germany,
        geo_spain
    ]

    prediction = model.predict([np.array(features)])
    result = "Customer is likely to churn" if prediction[0] == 1 else "Customer is likely to stay"

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
