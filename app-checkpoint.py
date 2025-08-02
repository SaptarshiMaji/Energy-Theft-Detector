from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load and train the model (or load from .pkl)
df = pd.read_csv("energy_data.csv")
df.drop(columns=['0'], inplace=True)

from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
df['Class'] = label_enc.fit_transform(df['Class'])
df['Theft'] = label_enc.fit_transform(df['Theft'])

X = df.drop(columns='Theft')
y = df['Theft']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

@app.route('/')
def form():
    with open("index.html", "r") as f:
        return f.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['electricity']),
            float(request.form['cooling']),
            float(request.form['heating']),
            float(request.form['equipment']),
            float(request.form['gas']),
            float(request.form['waterHeater'])
        ]
        while len(inputs) < X.shape[1]:
            inputs.append(0.0)

        input_scaled = scaler.transform([inputs])
        prediction = model.predict(input_scaled)[0]

        result = "Theft" if prediction != 0 else "Normal"
        return f"<h2>Prediction: {result}</h2><p>Total Input: {sum(inputs):.2f} kW</p><a href='/'>Go Back</a>"
    except Exception as e:
        return f"<p>Error: {e}</p><a href='/'>Try Again</a>"

if __name__ == '__main__':
    app.run(debug=True)
