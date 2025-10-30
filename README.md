⚡ VIDYUT — Smart Energy Consumption Predictor

Predict. Save. Sustain.
An AI-powered solution to analyze, predict, and optimize campus electricity usage.

🧩 Overview

Vidyut is an end-to-end AI system that predicts electricity consumption for campus buildings, detects anomalies (sudden spikes), and provides actionable suggestions via a dashboard.

It uses the ASHRAE / Kaggle building energy dataset, weather data, and engineered features (including a simulated occupancy feature) to produce building-level and campus-level energy forecasts.

This README explains the project architecture, workflow, dataset manipulations, model training, and how to run or demo the system.

🔥 Key Highlights — What Makes VIDYUT Unique

🏫 Campus-Centric Design: Works per building and aggregates to campus level.

🌦️ Multi-Source Input: Combines weather, building metadata, and simulated occupancy.

⚙️ ML-Powered: Uses RandomForest / XGBoost within a preprocessing → pipeline setup.

🚨 Anomaly Detection: Identifies sudden consumption spikes and offers suggestions.

💡 Lightweight Deployment: Model served via Flask API + simple dashboard (Streamlit / HTML).

⚡ Scalable: Can extend easily to real-time IoT sensor integration.



You can explore the live demo of Vidyut hosted on Vercel, which showcases real-time energy predictions, building comparisons, and anomaly detection — all powered by our trained ML model.

🌐 https://vidyut-4fom.vercel.app/

👉 Open the Live App on Vercel

(replace the above with your actual Vercel link)

🧭 What You Can Do in the Demo
Section	Description
🏠 Home Dashboard	Overview of predicted energy usage for all campus buildings.
📊 Building Insights	Select any building to view its hourly and daily energy trends.
⚙️ Prediction Panel	Enter conditions (temperature, humidity, occupancy, etc.) and instantly get predicted energy consumption.
🚨 Anomaly Alerts	See highlighted points when predicted usage exceeds normal thresholds.
🌡️ Feature Impact View	View which factors (like temperature, time, or occupancy) most affect predictions.
💡 Suggestions Panel	Get energy optimization recommendations (e.g., “Shift lab load to off-peak hours”).



📁 Repository Structure
VIDYUT/
├── data/                           # Raw / downloaded files
│   ├── train.csv
│   ├── weather_train.csv
│   └── building_metadata.csv
├── notebooks/
│   ├── 01_data_prep.ipynb          # Preprocessing & feature engineering
│   └── 02_training_evaluation.ipynb
├── src/
│   ├── preprocess.py               # Merge + create final_project_dataset.csv
│   ├── train_model.py              # Train models and save best model (energy_model.joblib)
│   ├── infer.py                    # Inference helper (loads model, returns predictions)
│   └── anomaly.py                  # Anomaly detection helpers
├── backend/
│   ├── app.py                      # Flask API for predictions & dashboard
│   └── requirements.txt
├── frontend/
│   └── (static dashboard files)    # Streamlit / React / Plotly dashboard
├── models/
│   └── energy_model.joblib         # Trained model
├── outputs/
│   └── test_predictions.csv
├── README.md
└── LICENSE



📊 Dashboard & Demo
🖥️ Visual Sections

Energy Trends — Line chart for daily/weekly/monthly energy usage

Building Comparison — Bar chart comparing energy across buildings

Anomaly Alerts — Highlighted spikes above threshold

Feature Importance — Display of most influential factors

Smart Suggestions — Automatic optimization tips

🧭 Implementation Options

Streamlit (quick prototype)

Plotly Dash / React + Chart.js (production-ready)

🧮 Data Manipulations (Feature Engineering Summary)
Step	Operation	Description
1️⃣	Merge Datasets	Combine train + weather + metadata
2️⃣	Time Features	Extract hour, weekday, month, and custom time_of_day
3️⃣	Occupancy Simulation	Estimate no_of_people from building type, time, and size
4️⃣	Encoding	One-hot encode building_type, time_of_day, etc.
5️⃣	Anomaly Detection	Flag predicted_energy_kWh > mean + 2*std
6️⃣	Visualization	Matplotlib / Plotly for trends, comparison, and anomalies
🧠 Design Decisions

Occupancy Simulation: Rule-based, explainable, mimics real human activity patterns.

Model Choice: RandomForest / XGBoost — handles mixed data types, interpretable via feature importances.

Anomaly Detection: Threshold-based for prototype simplicity; can upgrade to Isolation Forest or LSTM.

🔮 Future Enhancements

🔗 Real IoT integration via MQTT for real-time energy tracking.

⏳ LSTM-based time series forecasting for multi-step prediction.

🌞 Integration with solar generation & carbon footprint analysis.

🧾 Auto-report generator with PDF insights.

♻️ Cloud-based deployment for multi-campus monitoring.

🧑‍💻 Team & Credits

Developed by Team VIDYUT — 

 Lakshuki Hatwar
 Siddhi Dhoke
 Ness Dubey

Datasets: ASHRAE / Kaggle Energy Prediction Competition

Libraries: scikit-learn, pandas, Flask, Plotly, XGBoost




