⚡ Energy Theft Detector

A Flask web application that uses machine learning to detect potential energy theft based on electricity, gas, and equipment usage patterns. Built with Python, scikit-learn, and Random Forest classification.

🚀 Features

* Predicts possible energy theft based on user input.
* Trained on real consumption data (`energy_data.csv`).
* User-friendly HTML interface with clean UI.
* Backend powered by Flask and scikit-learn.
* Automatically scales features and supports additional inputs.

🏗️ Project Structure

energy-theft-detector/
├── app.py                  # Flask backend and ML pipeline
├── energy_data.csv         # Training dataset
├── model.pkl               # Saved trained model (auto-generated)
├── scaler.pkl              # Saved scaler (auto-generated)
├── templates/
│   ├── index.html          # Form for user input
│   └── result.html         # Result display page
└── README.md               # Project documentation


🔧 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/energy-theft-detector.git
cd energy-theft-detector
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Example `requirements.txt`:

```
flask
numpy
pandas
scikit-learn
joblib
```

3. **Run the application**

```bash
python app.py
```
Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

🧠 Model
* Algorithm: RandomForestClassifier
* Features: Electricity, Cooling, Heating, Equipment, Gas, Water Heater usage
* Scaled with `StandardScaler`
* Labels encoded using `LabelEncoder`

📸 Screenshots

<img width="1143" height="722" alt="image" src="https://github.com/user-attachments/assets/92c3d061-3b27-41a7-81de-ef3a7ed1be9b" />

📬 Contact
For questions or support, reach out to saptarshimaji10@gmail.com 

⭐ GitHub Topics
`machine-learning` `flask` `energy` `fraud-detection` `web-app` `scikit-learn` `energy-theft`
