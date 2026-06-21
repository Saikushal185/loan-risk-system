import pandas as pd
from src.validation import validate


def test_valid_frame_passes():
    df = pd.DataFrame({c: [1] for c in
                       ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                        "Loan_Amount_Term", "Credit_History"]})
    assert validate(df) == []


def test_missing_column_detected():
    df = pd.DataFrame({"ApplicantIncome": [1]})
    assert any("missing" in p for p in validate(df))
