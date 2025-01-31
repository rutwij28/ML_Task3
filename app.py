from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler safely
model_path = "titanic_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file is missing!")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route to avoid 404 errors
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Titanic Prediction API! Use POST /predict to get predictions."})

# Handle favicon.ico requests (prevents unnecessary errors in logs)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content response

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure "features" exist in request
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400
        
        features = np.array(data["features"]).reshape(1, -1)
        features = scaler.transform(features)  # Apply scaling

        prediction = model.predict(features)
        result = "Survived" if prediction[0] == 1 else "Not Survived"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
