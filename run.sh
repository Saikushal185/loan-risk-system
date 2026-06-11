#!/usr/bin/env bash
# Train the risk model if needed, then launch the dashboard.
set -e
cd "$(dirname "$0")"
if [ ! -f models/loan_risk_model.joblib ]; then
    python3 src/eda.py
    python3 src/train.py
fi
python3 -m streamlit run dashboard/app.py
