import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Correct the model path
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load the trained model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_array = np.array([features])
    
    prediction = model.predict(features_array)
    
    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    
    return render_template('index.html', prediction_text=f'The person is likely {output}')

if __name__ == '__main__':
    app.run(debug=True)
