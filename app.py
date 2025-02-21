from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Root route to check if API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "KOOS Prediction API is running!"})

# Load trained model at startup (ensure file exists)
model_path = os.getenv("MODEL_PATH", "KOOS_prediction_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"✅ Loading model from {model_path}...")
model = joblib.load(model_path)

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

        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            print("❌ ERROR: Features must be a list of numbers!")
            return jsonify({"error": "Invalid input format"}), 400

        # Predict using the preloaded model
        prediction = model.predict([features])
        print("📊 Prediction:", prediction)  # Log prediction result

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("🔥 ERROR:", e)  # Log any errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
