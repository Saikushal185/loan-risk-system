"""EDA plots: risk distributions and default trends.

Run from the project root:
    python3 src/eda.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import clean_data, engineer_features, load_data

PLOTS = ROOT / "reports" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def main() -> None:
    df = engineer_features(clean_data(load_data()))
    df["Default"] = (df["Loan_Status"] == "N").map({True: "Default", False: "Repaid"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x="TotalIncome", hue="Default", bins=40, ax=axes[0])
    axes[0].set_title("Income distribution by outcome")
    sns.histplot(data=df, x="LoanToIncome", hue="Default", bins=40, ax=axes[1])
    axes[1].set_title("Loan-to-income ratio by outcome")
    fig.savefig(PLOTS / "risk_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Credit_History", "Property_Area", "Education"]):
        rates = df.groupby(col)["Loan_Status"].apply(lambda s: (s == "N").mean())
        rates.plot.bar(ax=ax, color="#d62728")
        ax.set_title(f"Default rate by {col}")
        ax.set_ylabel("Default rate")
    fig.savefig(PLOTS / "default_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("EDA plots saved to reports/plots/")


if __name__ == "__main__":
    main()
