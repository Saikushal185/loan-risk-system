"""Data preparation for the loan default risk model.

Target: default = 1 when Loan_Status == 'N' (loan not approved / risky),
so the model outputs a probability of *default risk*.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("loan")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "loan_train.csv"

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
               "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
           "Loan_Amount_Term", "Credit_History"]
ENGINEERED = ["TotalIncome", "LoanToIncome", "EMI", "BalanceIncome"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d loan applications", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (mode for categoricals, median for numerics)."""
    df = df.drop(columns=["Loan_ID"]).copy()
    for col in CATEGORICAL:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
        df[col] = df[col].fillna(df[col].median())

    # Cap extreme income/loan outliers at the 99th percentile.
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk-relevant ratios used by lenders."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanToIncome"] = df["LoanAmount"] * 1000 / df["TotalIncome"].replace(0, np.nan)
    df["LoanToIncome"] = df["LoanToIncome"].fillna(df["LoanToIncome"].median())
    df["EMI"] = df["LoanAmount"] * 1000 / df["Loan_Amount_Term"]
    df["BalanceIncome"] = df["TotalIncome"] - df["EMI"]
    return df


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode and split features/target (1 = default risk)."""
    y = (df["Loan_Status"] == "N").astype(int)
    X = pd.get_dummies(df.drop(columns=["Loan_Status"]),
                       columns=CATEGORICAL, drop_first=True)
    return X, y
