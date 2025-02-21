Python 3.12.9 (tags/v3.12.9:fdb8142, Feb  4 2025, 15:27:58) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from flask import Flask, request, jsonify
... import joblib
... import numpy as np
... 
... app = Flask(__name__)
... 
... # Load trained model (make sure you have 'model.pkl' saved)
... model = joblib.load("KOOS_prediction_model.pkl")
... 
... @app.route('/predict', methods=['POST'])
... def predict():
...     data = request.get_json()
...     features = np.array(data['features']).reshape(1, -1)
...     prediction = model.predict(features)
...     return jsonify({'prediction': prediction.tolist()})
... 
... if __name__ == '__main__':
...     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
