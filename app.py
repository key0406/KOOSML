from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
 
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)  # Log received input

        if not data or "features" not in data:
            print("❌ ERROR: Missing 'features' in request!")
            return jsonify({"error": "Invalid input"}), 400

        features = data["features"]
        print("🧩 Extracted Features:", features)  # Log features

        # Load ML Model
        import joblib
        model = joblib.load("model.pkl")  # Ensure this file exists!

        prediction = model.predict([features])
        print("📊 Prediction:", prediction)  # Log prediction result

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("🔥 ERROR:", e)  # Log any errors
        return jsonify({"error": str(e)}), 500

