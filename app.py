from flask import Flask, request, jsonify
import joblib
import os
import xgboost as xgb
import numpy as np
import pandas as pd  # Required for feature alignment

app = Flask(__name__)

# Load trained model at startup
model_path = os.getenv("MODEL_PATH", "KOOS_prediction_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"✅ Loading model from {model_path}...")
model = joblib.load(model_path)

# Define expected feature columns
EXPECTED_FEATURES = ["P1", "P2", "P3", "P4", "f1", "f2", "f3", "f4", "q1", "q2", "q3", "q4", "Pain", "Function", "QOL"]

@app.route('/predict', methods=['PUT'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)

        if data or "features":
            return jsonify({"message": "Fetched Successfully"}),200

        if not data or "features" not in data:
            return jsonify({"error": "Invalid input format, 'features' missing"}), 400

        # Convert input dictionary to Pandas DataFrame
        df = pd.DataFrame([data["features"]])  

        # Ensure feature order matches training data
        if set(EXPECTED_FEATURES) - set(df.columns):
            missing = list(set(EXPECTED_FEATURES) - set(df.columns))
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        # Convert to XGBoost DMatrix
        features_matrix = xgb.DMatrix(df[EXPECTED_FEATURES])

        # Predict
        prediction = model.predict(features_matrix)
        print("📊 Prediction:", prediction)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)