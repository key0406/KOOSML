from flask import Flask, request, jsonify
import joblib
import os
import xgboost as xgb
import numpy as np
import pandas as pd  # Required for feature alignment

app = Flask(__name__)

# Load trained model at startup
model_path = os.getenv("MODEL_PATH", "KOOS_prediction_3500_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"✅ Loading model from {model_path}...")
model = joblib.load(model_path)

# Define expected feature columns
EXPECTED_FEATURES = ["P1", "P2", "P3", "P4", "f1", "f2", "f3", "f4", "q1", "q2", "q3", "q4", "Pain", "Function", "QOL"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)  # Debugging

        if not data or "features" not in data:
            return jsonify({"error": "Invalid input format, 'features' missing"}), 400

        df = pd.DataFrame([data["features"]])  

        missing = list(set(EXPECTED_FEATURES) - set(df.columns))
        if missing:
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        features_matrix = xgb.DMatrix(df[EXPECTED_FEATURES])
        prediction = model.predict(features_matrix)

        print("📊 Prediction:", prediction)
        return jsonify({"prediction": prediction.tolist()})  # ✅ Ensure this is returned

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)