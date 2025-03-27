from flask import Flask, request, jsonify
import joblib
import pandas as pd
import yfinance as yf
import os
import numpy as np  

app = Flask(__name__)

# Load trained model safely
MODEL_PATH = "coca_cola_model.pkl"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    model = None
else:
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "Welcome to the Stock Prediction API! Use /predict to get predictions."

@app.route('/predict', methods=['GET'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model is not loaded."}), 500

        # Fetch real-time stock data
        ticker = "KO"
        live_data = yf.download(ticker, period='5d', interval='1h')

        # Debugging: Print available columns
        print("Fetched data columns:", live_data.columns.tolist())

        # Check if data is available
        if live_data.empty:
            return jsonify({"error": "No stock data available. Try again later."}), 500

        # Extract latest features safely
        latest_data = live_data.iloc[-1].fillna(0)  # Fill NaN values with 0

       
        numeric_features = [
            float(latest_data['Open']),
            float(latest_data['High']),
            float(latest_data['Low']),
            float(latest_data['Volume']),
            float(latest_data['Close']),  
            float(latest_data['Close']), 
            0.0  # Placeholder feature
        ]

        features = np.array(numeric_features, dtype=np.float64).reshape(1, -1)

        print(f"Features being passed to model: {features}")  # Debugging log

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({"Predicted Closing Price": float(prediction)})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
