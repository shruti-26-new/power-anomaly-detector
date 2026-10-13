   # Power Consumption Anomaly Detector

   Detects anomalies in SCADA/PMU-style power data using Isolation Forest.
   Prototype built on synthetic data.

   ## Tech stack
   - Backend: FastAPI, SQLite (SQL), pandas, scikit-learn
   - Frontend: React, Chart.js

   ## How it works
   1. /generate creates synthetic voltage, current, frequency and power readings with injected faults.
   2. /analyze trains an Isolation Forest and flags anomalies in the database.
   3. The React dashboard plots the readings and highlights anomalies.

   ## Run locally
   Backend: cd backend, pip install -r requirements.txt, uvicorn main:app --reload
   Frontend: cd frontend, python -m http.server 5500, then open http://localhost:5500
