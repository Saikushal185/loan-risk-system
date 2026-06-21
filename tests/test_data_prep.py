import pandas as pd
from src import data_prep


def test_clean_data_has_no_missing():
    df = data_prep.load_data()
    cleaned = data_prep.clean_data(df)
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_drops_loan_id():
    df = data_prep.load_data()
    cleaned = data_prep.clean_data(df)
    assert "Loan_ID" not in cleaned.columns
