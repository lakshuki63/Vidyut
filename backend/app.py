import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

# 1. Initialize the Flask App
app = Flask(__name__)
CORS(app) # This allows your frontend to talk to your backend

# 2. Load Your Trained Model
try:
    pipeline = joblib.load('energy_model.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'energy_model.joblib' file not found.")
    pipeline = None

# 3. Define the Prediction & Anomaly Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        # Get all JSON data from the frontend
        data = request.get_json()

        # --- ANOMALY LOGIC (NEW) ---
        
        # 1. Get the "actual" usage sent from the frontend
        # We .pop() it because it's not needed for the prediction itself
        if 'actual_energy_kWh' not in data:
            return jsonify({'error': 'Missing required field: actual_energy_kWh'}), 400
            
        actual_usage = data.pop('actual_energy_kWh')

        # 2. Prepare the rest of the data for the prediction model
        input_data = pd.DataFrame([data])
        
        # 3. Make the *predicted* usage
        predicted_usage = pipeline.predict(input_data)[0]

        # 4. Compare actual vs. predicted to find anomalies
        is_anomaly = False
        anomaly_message = "✅ Usage is within expected range."

        # We define an anomaly as 50% higher than predicted
        anomaly_threshold = predicted_usage * 1.5

        if actual_usage > anomaly_threshold and actual_usage > 5: # (and over 5 kWh to avoid 0 vs 1)
            is_anomaly = True
            anomaly_message = f"⚠️ Anomaly Alert! Actual use ({actual_usage} kWh) is 50%+ higher than predicted ({predicted_usage:.2f} kWh)."

        # --- END OF ANOMALY LOGIC ---

        # 5. Send the full response back
        return jsonify({
            'predicted_energy_kWh': round(predicted_usage, 2),
            'actual_energy_kWh': actual_usage,
            'is_anomaly': is_anomaly,
            'anomaly_message': anomaly_message
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# 6. Run the App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)