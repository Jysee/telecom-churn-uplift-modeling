from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR


def score_customers(
    artifact_path: Path, input_path: Path, output_path: Path
) -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    frame = pd.read_csv(input_path)
    feature_columns: list[str] = artifact["feature_columns"]
    missing = sorted(set(feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    X = frame.loc[:, feature_columns].copy()
    for column in artifact["categorical_features"]:
        X[column] = X[column].astype(str)

    model = artifact["model"]
    p0, p1 = model.predict_potential_outcomes(X)
    score = model.predict_score(X)
    fraction = float(artifact["policy_fraction"])
    recommendation = np.zeros(len(X), dtype=bool)
    if fraction > 0:
        k = max(1, int(np.ceil(len(X) * fraction)))
        recommendation[np.argsort(-score, kind="mergesort")[:k]] = True

    result = pd.DataFrame(
        {
            "p_churn_without_treatment": p0,
            "p_churn_with_treatment": p1,
            "ranking_score": score,
            "recommended_treatment": recommendation,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a batch of telecom customers")
    parser.add_argument(
        "--model", type=Path, default=MODELS_DIR / "best_model.joblib"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_customers(args.model, args.input, args.output)
    print(f"Saved {len(result)} scored rows to {args.output}")


if __name__ == "__main__":
    main()
