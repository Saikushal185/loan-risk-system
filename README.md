# Loan Default Risk Prediction System

Explainable loan default risk scoring on the Loan Prediction dataset: three models compared, SHAP explanations, and a Streamlit approval-simulation dashboard.

## How to run
```bash
pip install -r requirements.txt
python3 src/eda.py                # risk distribution / default trend plots
python3 src/train.py              # train 3 models, SHAP, confusion matrices
streamlit run dashboard/app.py    # approval simulator with explanations
```
Or simply `./run.sh`.

## Project structure
```
loan-risk-system/
├── data/loan_train.csv          # Loan Prediction dataset (614 applications)
├── src/
│   ├── data_prep.py             # imputation, outlier capping, encoding
│   │                            #   engineered: TotalIncome, LoanToIncome, EMI, BalanceIncome
│   ├── eda.py                   # risk distributions + default trends
│   ├── train.py                 # LogReg / RF / XGBoost + metrics + SHAP
│   └── risk_engine.py           # risk bands + per-applicant SHAP explanation
├── dashboard/app.py             # input form, approval simulation, explanation
├── models/loan_risk_model.joblib
├── reports/                     # model_comparison.csv + plots/
├── notebooks/
├── requirements.txt
└── README.md
```

## Risk policy
| Default probability | Band | Decision |
|---|---|---|
| < 30% | Low risk | Approve |
| 30–55% | Medium risk | Approve with conditions |
| > 55% | High risk | Decline or require guarantor |

## Business recommendations
- Credit history is the dominant risk driver — verify it first; applicants without history should follow the conditional-approval path.
- High loan-to-income ratios drive defaults: cap LTI or require co-applicants above the threshold.
- Use the SHAP explanation in the dashboard to document every lending decision (audit trail).
