"""Centralised configuration for risk thresholds and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_train.csv"
MODEL_PATH = ROOT / "models" / "loan_risk_model.joblib"

# Probability thresholds for the three risk bands.
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.55

RANDOM_STATE = 42
TEST_SIZE = 0.2
