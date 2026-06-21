"""Lightweight schema validation for incoming applicant data."""
import pandas as pd

REQUIRED = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History"]


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation problems (empty means valid)."""
    problems = []
    for col in REQUIRED:
        if col not in df.columns:
            problems.append(f"missing required column: {col}")
    if "ApplicantIncome" in df and (df["ApplicantIncome"] < 0).any():
        problems.append("ApplicantIncome contains negative values")
    if "Credit_History" in df:
        bad = ~df["Credit_History"].dropna().isin([0, 1])
        if bad.any():
            problems.append("Credit_History must be 0 or 1")
    return problems
