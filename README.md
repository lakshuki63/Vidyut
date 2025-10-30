VIDYUT — Smart Energy Consumption Predictor ⚡️

Predict. Save. Sustain.

Vidyut is an end-to-end AI system that predicts electricity consumption for campus buildings, detects anomalies (sudden spikes), and provides actionable suggestions via a dashboard. It uses the ASHRAE / Kaggle building energy dataset, weather data, and engineered features (including a simulated occupancy feature) to produce building-level and campus-level energy forecasts.

This README explains the project, how to reproduce the results, run the model and API, and what to show in a demo or submission.

🔥 Key Highlights (what makes VIDYUT unique)

Combines weather, building metadata, time features, and (simulated) occupancy to predict energy usage.

Trains a robust tree-based model (RandomForest / XGBoost) in a preprocessing → pipeline workflow.

Offers anomaly detection (predicted spike vs typical range) and energy-saving suggestions.

Lightweight prototype: model saved as a joblib file and served via a Flask API + simple dashboard (HTML/JS or Streamlit).

Designed for campuses: per-building forecasts + combined campus forecasts (easy to scale).

📁 Repository structure (suggested)
VIDYUT/
├── data/                           # Raw / downloaded files (large files excluded from repo)
│   ├── train.csv
│   ├── weather_train.csv
│   └── building_metadata.csv
├── notebooks/
│   ├── 01_data_prep.ipynb          # Preprocessing & feature engineering (use Colab)
│   └── 02_training_evaluation.ipynb
├── src/
│   ├── preprocess.py               # Script: merge + create final_project_dataset.csv
│   ├── train_model.py              # Train models and save best model (energy_model.joblib)
│   ├── infer.py                    # Simple inference helper (loads model, returns preds)
│   └── anomaly.py                  # Anomaly detection helper functions
├── backend/
│   ├── app.py                      # Flask API for predictions & dashboard data
│   └── requirements.txt
├── frontend/
│   └── (static dashboard files)    # or Streamlit / React app
├── models/
│   └── energy_model.joblib         # Trained model (not committed if large)
├── outputs/
│   └── test_predictions.csv
├── README.md
└── LICENSE

🧾 What’s in this project (quick)

final_project_dataset.csv — cleaned, merged dataset used for training (timestamp, building_id, energy_kWh, temperature, humidity, no_of_people, day_of_week, time_of_day, building_type, square_feet)

train_model.py — trains several models (RandomForest, ExtraTrees, XGBoost), compares R²/MAE/RMSE, saves the best pipeline to models/energy_model.joblib

backend/app.py — Flask API endpoints for prediction, dashboard data, and anomalies

notebooks/ — reproducible Colab/Notebook workflow for preprocessing, training, evaluation

frontend/ — code or template for dashboard visualizations (Plotly / Chart.js / Streamlit)

🧰 Requirements / Environment

Create an environment and install dependencies. Example backend/requirements.txt:

pandas
numpy
scikit-learn
xgboost
joblib
flask
matplotlib
plotly
seaborn
python-dateutil


Install:

python -m venv venv
source venv/bin/activate      # on Windows use: venv\Scripts\activate
pip install -r backend/requirements.txt

⚙️ Step 1 — Prepare data (COLAB recommended)

The Kaggle ASHRAE dataset is large. We recommend doing heavy preprocessing & training in Google Colab (or a machine with sufficient RAM).

Notebook / Script workflow

Download the Kaggle files and place them in data/ (train.csv, weather_train.csv, building_metadata.csv).

Run notebooks/01_data_prep.ipynb or src/preprocess.py. This will:

Filter meter == 0 (electricity)

Merge weather and meta data

Create time features (hour, day_of_week, time_of_day)

Create no_of_people (rule-based, scaled by square_feet and time-of-day)

Save final_project_dataset.csv

Example (simplified) command:

python src/preprocess.py --input_dir data/ --output final_project_dataset.csv

⚙️ Step 2 — Train model (Colab or local)

Use notebooks/02_training_evaluation.ipynb or src/train_model.py to:

Sample/Downsample if memory is constrained

Define pipeline: ColumnTransformer (passthrough numeric, OneHotEncoder categorical) + RandomForestRegressor (or XGBoost)

Train and evaluate (train/val split or cross-validation)

Save best pipeline (including preprocessor) with joblib.dump to models/energy_model.joblib

Simple example call:

python src/train_model.py --data final_project_dataset.csv --out models/energy_model.joblib


Typical evaluation metrics to report:

R² (coefficient of determination)

MAE (mean absolute error)

RMSE (root mean squared error)

⚙️ Step 3 — Test / Predict on final_test_dataset.csv

Preprocess test rows same as training (same features and encodings). Load the saved pipeline and predict.

Example helper snippet:

import joblib, pandas as pd
model = joblib.load("models/energy_model.joblib")
test_df = pd.read_csv("final_test_dataset.csv")
# ensure the same features exist and are encoded the same way, handle NaNs
X_test = test_df[feature_columns].dropna()
y_pred = model.predict(X_test)
test_df.loc[X_test.index, 'predicted_energy_kWh'] = y_pred
test_df.to_csv("outputs/test_predictions.csv", index=False)

🚀 Step 4 — Run Flask API (serve model in backend)

A minimal Flask API (backend/app.py) exposes endpoints like:

POST /predict
Request JSON: { "building_id": 3, "timestamp":"2025-10-30 14:00:00", "temperature":29.5, "humidity":60, "no_of_people":45, ... }
Response JSON: { "predicted_energy_kWh": 86.4 }

GET /dashboard-data?start=...&end=... — returns aggregated time series for frontend

Run:

cd backend
export FLASK_APP=app.py
flask run


(Windows: set FLASK_APP=app.py then flask run)

Sample curl:

curl -X POST http://127.0.0.1:5000/predict \
 -H "Content-Type: application/json" \
 -d '{"building_id":3,"timestamp":"2017-01-01 00:00:00","temperature":25,"humidity":45,"no_of_people":30,"square_feet":7432,"day_of_week":"Monday","time_of_day":"Morning","building_type":"Education"}'

📊 Dashboard & Demo

The frontend shows:

Per-building time series (actual vs predicted)

Combined campus forecast

Average predicted energy by building type

Anomalies highlighted (predicted > mean + k*std)

Suggestions panel (auto-generated tips e.g., “Shift heavy lab operations to off-peak hours”)

You can implement dashboard in:

Streamlit (very fast) or

React + Chart.js/Plotly (production-like)





🧾 Notes & design decisions

Occupancy (no_of_people) is simulated using a simple, explainable rule-based function (scales with square_feet, time_of_day, and building_type). This is a pragmatic choice when real occupancy sensors aren’t available — and it adds behavioral context to predictions.

Model choice: RandomForest (or XGBoost) is robust for mixed categorical + numeric tabular data and gives feature importances for explainability.

Anomaly detection: simple threshold method (predicted > mean + 2*std) for prototype; can be replaced with IsolationForest / One-Class SVM for more sophistication.

🛠️ Extending Vidyut (future & ideas)

Integrate real IoT sensor feeds (MQTT) for occupancy and meter readings.

Use time-series deep learning (LSTM) for sequence forecasting and multi-step ahead predictions.

Add cost and carbon computations (convert kWh to ₹ and CO₂).

Automatic report generator (PDF) and schedule-based actions (e.g., send email alert to facility manager).

Implement model retraining pipeline (daily/weekly) with newly collected data.

📬 Contact / Credits

Project & code by: 
Lakshuki Hatwar 
Siddhi Dhoke
Ness Dubey 
Datasets: ASHRAE / Kaggle Energy Prediction competition (link & credit in slides).
Open-source libraries: scikit-learn, XGBoost, pandas, Flask, Plotly.
