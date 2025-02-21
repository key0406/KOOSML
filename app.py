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
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
