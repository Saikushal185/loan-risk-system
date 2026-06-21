"""Score a CSV of applicants in bulk and write decisions to disk."""
import argparse
from pathlib import Path

import pandas as pd

from src import data_prep, risk_engine


def main(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)
    df = data_prep.engineer_features(data_prep.clean_data(df))
    bundle = risk_engine.load_bundle()
    rows = []
    for _, applicant in df.iterrows():
        result = risk_engine.score(bundle, applicant.to_frame().T)
        rows.append({"probability": result["probability"],
                     "band": result["band"], "decision": result["decision"]})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} decisions to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default="reports/batch_decisions.csv")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    main(args.input, args.output)
