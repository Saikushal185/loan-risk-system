"""Streamlit Loan Risk Dashboard: approval simulation with explainable scoring.

Run from the project root (after `python3 src/train.py`):
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import CATEGORICAL, engineer_features
from risk_engine import MODEL_PATH, load_bundle, score

st.set_page_config(page_title="Loan Risk", page_icon="🏦", layout="wide")
st.title("🏦 Loan Default Risk Prediction System")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python3 src/train.py` first.")
    st.stop()


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()
st.caption(f"Scoring model: {bundle['name']}")

with st.form("applicant"):
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    married = c1.selectbox("Married", ["Yes", "No"])
    dependents = c1.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = c2.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = c2.selectbox("Self employed", ["No", "Yes"])
    area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    income = c3.number_input("Applicant income (monthly)", 0, 100_000, 5_000)
    co_income = c3.number_input("Co-applicant income (monthly)", 0, 100_000, 0)
    amount = c3.number_input("Loan amount (thousands)", 1, 1_000, 120)
    term = c1.selectbox("Term (months)", [360, 180, 240, 120, 300, 60], index=0)
    credit = c2.selectbox("Has credit history?", ["Yes", "No"])
    submitted = st.form_submit_button("Simulate approval", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_emp,
        "ApplicantIncome": income, "CoapplicantIncome": co_income,
        "LoanAmount": amount, "Loan_Amount_Term": term,
        "Credit_History": 1.0 if credit == "Yes" else 0.0, "Property_Area": area,
    }])
    row = engineer_features(row)
    row = pd.get_dummies(row, columns=CATEGORICAL)
    result = score(bundle, row)

    k1, k2, k3 = st.columns(3)
    k1.metric("Default probability", f"{result['probability']:.1%}")
    k2.metric("Risk band", result["band"])
    k3.metric("EMI", f"{float(row['EMI'].iloc[0]):,.0f}/month")

    if result["band"] == "Low risk":
        st.success(f"**Decision: {result['decision']}**")
    elif result["band"] == "Medium risk":
        st.warning(f"**Decision: {result['decision']}**")
    else:
        st.error(f"**Decision: {result['decision']}**")

    st.subheader("Why? — top risk contributions (SHAP)")
    contrib = result["contributions"].rename("Impact on risk").to_frame()
    contrib["Direction"] = ["increases risk" if v > 0 else "decreases risk"
                            for v in contrib["Impact on risk"]]
    st.bar_chart(contrib["Impact on risk"])
    st.dataframe(contrib.round(4), use_container_width=True)

st.divider()
st.subheader("Model evaluation artefacts")
comp = ROOT / "reports" / "model_comparison.csv"
if comp.exists():
    st.dataframe(pd.read_csv(comp), use_container_width=True)
for img in ["feature_importance.png", "shap_summary.png", "risk_distributions.png"]:
    p = ROOT / "reports" / "plots" / img
    if p.exists():
        st.image(str(p))
