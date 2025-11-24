
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('blood_donation_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['Recency'], data['Frequency'], data['Monetary'], data['Time']]).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = int(model.predict(scaled)[0])
    return jsonify({"Prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
