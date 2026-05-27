"""Train and evaluate loan default models; produce explainability artefacts.

Run from the project root:
    python3 src/train.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import clean_data, encode, engineer_features, load_data, log

PLOTS = ROOT / "reports" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000,
                                                  class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                class_weight="balanced",
                                                random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 scale_pos_weight=2.2, eval_metric="logloss",
                                 random_state=42),
    }


def main() -> None:
    df = engineer_features(clean_data(load_data()))
    X, y = encode(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    results, fitted = [], {}
    for name, model in build_models().items():
        # Linear model needs scaling; trees don't.
        if name == "Logistic Regression":
            model.fit(Xtr_s, y_train)
            proba = model.predict_proba(Xte_s)[:, 1]
            pred = model.predict(Xte_s)
        else:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = model.predict(X_test)
        fitted[name] = model
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Repaid", "Default"], cmap="Blues")
        plt.title(f"Confusion matrix — {name}")
        plt.savefig(PLOTS / f"cm_{name.lower().replace(' ', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        log.info("%s: AUC=%.3f F1=%.3f", name, results[-1]["ROC-AUC"], results[-1]["F1"])

    res = pd.DataFrame(results).round(4)
    res.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    best_name = res.loc[res["ROC-AUC"].idxmax(), "Model"]
    best = fitted[best_name]
    log.info("Best model: %s", best_name)

    # ---- Explainability: feature importance + SHAP on the best tree model ----
    tree_name = best_name if best_name != "Logistic Regression" else "Random Forest"
    tree = fitted[tree_name]
    imp = pd.Series(tree.feature_importances_, index=X.columns).nlargest(12)
    imp.sort_values().plot.barh(figsize=(8, 6), color="#1f77b4")
    plt.title(f"Feature importance — {tree_name}")
    plt.tight_layout()
    plt.savefig(PLOTS / "feature_importance.png", dpi=150)
    plt.close()

    explainer = shap.TreeExplainer(tree)
    shap_values = explainer(X_test)
    if shap_values.values.ndim == 3:  # RF returns one set per class
        shap_values = shap_values[..., 1]
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.title("SHAP summary — risk drivers")
    plt.savefig(PLOTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": best, "name": best_name, "columns": list(X.columns),
                 "scaler": scaler, "needs_scaling": best_name == "Logistic Regression",
                 "tree_for_shap": tree},
                ROOT / "models" / "loan_risk_model.joblib")

    print(res.to_string(index=False))
    print(f"\nBest model ({best_name}) saved to models/loan_risk_model.joblib")


if __name__ == "__main__":
    main()
