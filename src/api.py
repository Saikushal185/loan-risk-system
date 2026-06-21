"""Minimal FastAPI service exposing the risk model."""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src import data_prep, risk_engine

app = FastAPI(title="Loan Risk API")
_bundle = None


class Applicant(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float = 0.0
    LoanAmount: float
    Loan_Amount_Term: float = 360.0
    Credit_History: float = 1.0
    Gender: str = "Male"
    Married: str = "Yes"
    Dependents: str = "0"
    Education: str = "Graduate"
    Self_Employed: str = "No"
    Property_Area: str = "Urban"


@app.on_event("startup")
def _load():
    global _bundle
    _bundle = risk_engine.load_bundle()


@app.post("/score")
def score(applicant: Applicant):
    df = pd.DataFrame([applicant.model_dump()])
    df = data_prep.engineer_features(data_prep.clean_data(
        df.assign(Loan_ID="api", Loan_Status="Y")))
    result = risk_engine.score(_bundle, df)
    return {"probability": result["probability"],
            "band": result["band"], "decision": result["decision"]}
