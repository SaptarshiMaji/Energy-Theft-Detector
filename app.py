# app.py
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Load and preprocess data only once
def train_and_save_model():
    df = pd.read_csv("energy_data.csv", encoding="latin-1")
    df.drop(columns=['0'], inplace=True, errors='ignore')

    label_enc = LabelEncoder()
    df['Class'] = label_enc.fit_transform(df['Class'])
    df['Theft'] = label_enc.fit_transform(df['Theft'])

    X = df.drop(columns='Theft')
    y = df['Theft']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return model, scaler, X.shape[1]

# Load or train model
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    df = pd.read_csv("energy_data.csv", encoding="latin-1")
    feature_count = df.drop(columns=['0', 'Theft'], errors='ignore').shape[1]
else:
    model, scaler, feature_count = train_and_save_model()

@app.route('/')
def form():
    return render_template("index.html")

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

        # Pad with zeros if the dataset has more features
        while len(inputs) < feature_count:
            inputs.append(0.0)

        input_scaled = scaler.transform([inputs])
        prediction = model.predict(input_scaled)[0]

        result = "Theft" if prediction != 0 else "Normal"
        return render_template("result.html", result=result, total=sum(inputs))
    except Exception as e:
        return render_template("result.html", result=f"Error: {e}", total=0)

if __name__ == '__main__':
    app.run(debug=True)
