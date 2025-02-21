from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
 
app = Flask(__name__)

# Root route to check if API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "KOOS Prediction API is running!"})

# Load trained model (make sure you have 'model.pkl' saved)
model_path = os.getenv("MODEL_PATH", "KOOS_prediction_model.pkl")

# モデルが存在するか確認してロード
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

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

        # Load ML Model
        import joblib
        model = joblib.load("model.pkl")  # Ensure this file exists!

        prediction = model.predict([features])
        print("📊 Prediction:", prediction)  # Log prediction result

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("🔥 ERROR:", e)  # Log any errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
