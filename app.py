from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Initialize Flask with absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Debug prints
print(f"\n=== DEBUG INFO ===")
print(f"Base directory: {base_dir}")
print(f"Template directory: {template_dir}")
print(f"Files in templates: {os.listdir(template_dir)}\n")

# Load model
try:
    model = joblib.load('heart_disease_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Encoding dictionaries
CHEST_PAIN_DICT = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
ECG_DICT = {'LVH': 0, 'Normal': 1, 'ST': 2}
ST_SLOPE_DICT = {'Down': 0, 'Flat': 1, 'Up': 2}
SEX_DICT = {'F': 0, 'M': 1}
ANGINA_DICT = {'N': 0, 'Y': 1}
FBS_DICT = {'Normal': 0, 'High': 1}

@app.route('/')
def home():
    """Main page that serves the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        print("\nReceived data:", data)
        
        # Validate required fields
        required_fields = ['age', 'sex', 'chest_pain', 'rbp', 'chol', 
                         'fbs', 'ecg', 'max_hr', 'angina', 'oldpeak', 'st_slope']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing fields'}), 400

        # Convert input data to proper types
        input_data = [
            int(data['age']),  # Age: int
            SEX_DICT[data['sex']],  # Sex: encoded int
            CHEST_PAIN_DICT[data['chest_pain']],  # ChestPainType: encoded int
            int(data['rbp']),  # RestingBP: int
            int(data['chol']),  # Cholesterol: int
            FBS_DICT[data['fbs']],  # FastingBS: encoded int
            ECG_DICT[data['ecg']],  # RestingECG: encoded int
            int(data['max_hr']),  # MaxHR: int
            ANGINA_DICT[data['angina']],  # ExerciseAngina: encoded int
            float(data['oldpeak']),  # Oldpeak: float
            ST_SLOPE_DICT[data['st_slope']]  # ST_Slope: encoded int
        ]
        # Create DataFrame with correct column names and types
        input_df = pd.DataFrame([input_data], columns=[
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
            'Oldpeak', 'ST_Slope'
        ])
        
        print("\nProcessed data for prediction:")
        print(input_df.dtypes)
        print(input_df)
        
        # Predict
        prob = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'probability': f"{prob*100:.1f}%",
            'risk_level': get_risk_level(prob),
            'prediction': int(model.predict(input_df)[0])
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def get_risk_level(prob):
    if prob < 0.3:
        return "LOW RISK - Keep up the healthy lifestyle!"
    elif prob < 0.7:
        return "MODERATE RISK - Consider regular checkups."
    return "HIGH RISK - Please consult a doctor immediately."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)