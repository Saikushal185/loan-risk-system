"""Risk scoring engine: turns model probability into a business decision
with a per-feature explanation (SHAP contributions)."""

from pathlib import Path

import joblib
import pandas as pd
import shap

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "loan_risk_model.joblib"

RISK_BANDS = [(0.30, "Low risk", "Approve"),
              (0.55, "Medium risk", "Approve with conditions (higher rate / collateral)"),
              (1.01, "High risk", "Decline or require guarantor")]


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def score(bundle: dict, applicant: pd.DataFrame) -> dict:
    """Score one applicant row (already feature-engineered, pre-encoding)."""
    X = pd.get_dummies(applicant, drop_first=False)
    X = X.reindex(columns=bundle["columns"], fill_value=0)

    feats = bundle["scaler"].transform(X) if bundle["needs_scaling"] else X
    prob = float(bundle["model"].predict_proba(feats)[0, 1])

    for threshold, band, decision in RISK_BANDS:
        if prob < threshold:
            break

    # SHAP explanation from the tree model (always tree-explainable).
    explainer = shap.TreeExplainer(bundle["tree_for_shap"])
    sv = explainer(X)
    values = sv.values[0] if sv.values.ndim == 2 else sv.values[0, :, 1]
    contrib = (pd.Series(values, index=bundle["columns"])
               .sort_values(key=abs, ascending=False).head(8))

    return {"probability": prob, "band": band, "decision": decision,
            "contributions": contrib}
