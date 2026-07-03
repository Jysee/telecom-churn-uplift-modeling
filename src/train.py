from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORT_PATH,
    BusinessConfig,
)
from src.data_preprocessing import (
    clean_data,
    create_features,
    get_feature_types,
    load_data,
    split_features_target,
)
from src.evaluation import (
    classification_metrics,
    evaluate_uplift,
    plot_calibration,
    plot_feature_importance,
    plot_model_comparison,
    plot_policy_value,
    plot_test_uplift_curve,
    plot_treatment_outcomes,
)
from src.models import CANDIDATE_NAMES, UpliftEstimator, build_candidate
from src.reporting import write_final_report
from src.uplift_metrics import bootstrap_uplift_metrics, choose_policy
from src.utils import ensure_output_directories, save_json, set_random_seed


def joint_strata(y: pd.Series, treatment: pd.Series) -> pd.Series:
    return treatment.astype(str) + "_" + y.astype(str)


def _factual_evaluation_slice(
    model_name: str,
    y: pd.Series,
    treatment: pd.Series,
    factual_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name.startswith("outcome_risk"):
        mask = treatment.to_numpy() == 0
        return y.to_numpy()[mask], factual_probability[mask]
    return y.to_numpy(), factual_probability


def cross_validate_candidate(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    *,
    n_splits: int,
    top_fraction: float,
    seed: int,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray], pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strata = joint_strata(y, treatment)
    score = np.zeros(len(X), dtype=float)
    p0 = np.zeros(len(X), dtype=float)
    p1 = np.full(len(X), np.nan, dtype=float)
    factual = np.zeros(len(X), dtype=float)

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X, strata), start=1
    ):
        X_train = X.iloc[train_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        t_train = treatment.iloc[train_indices].reset_index(drop=True)
        X_validation = X.iloc[validation_indices].reset_index(drop=True)
        t_validation = treatment.iloc[validation_indices].reset_index(drop=True)

        model = build_candidate(model_name, X_train, seed=seed + fold)
        model.fit(X_train, y_train, t_train)
        fold_p0, fold_p1 = model.predict_potential_outcomes(X_validation)
        p0[validation_indices] = fold_p0
        p1[validation_indices] = fold_p1
        score[validation_indices] = model.predict_score(X_validation)
        factual[validation_indices] = model.predict_factual(
            X_validation, t_validation
        )

    uplift_metrics, curve = evaluate_uplift(
        y.to_numpy(),
        treatment.to_numpy(),
        score,
        top_fraction=top_fraction,
    )
    factual_y, factual_probability = _factual_evaluation_slice(
        model_name, y, treatment, factual
    )
    predictive_metrics = classification_metrics(factual_y, factual_probability)
    metrics: dict[str, float | int | str] = {
        "model": model_name,
        **uplift_metrics,
        **predictive_metrics,
    }
    predictions = {
        "score": score,
        "p0": p0,
        "p1": p1,
        "factual_probability": factual,
    }
    return metrics, predictions, curve


def dataset_summary(
    X: pd.DataFrame, y: pd.Series, treatment: pd.Series
) -> dict[str, Any]:
    numeric, categorical = get_feature_types(X)
    control = treatment.eq(0)
    treated = treatment.eq(1)
    return {
        "rows": len(X),
        "features": X.shape[1],
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "churn_rate": float(y.mean()),
        "treatment_rate": float(treatment.mean()),
        "control_churn_rate": float(y.loc[control].mean()),
        "treated_churn_rate": float(y.loc[treated].mean()),
    }


def library_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "catboost",
        "joblib",
        "matplotlib",
        "seaborn",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seed(args.seed)
    ensure_output_directories()

    raw = load_data(args.data)
    clean = clean_data(raw)
    featured = create_features(clean)
    X, y, treatment = split_features_target(featured)

    all_indices = np.arange(len(X))
    development_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=joint_strata(y, treatment),
    )
    X_development = X.iloc[development_indices].reset_index(drop=True)
    y_development = y.iloc[development_indices].reset_index(drop=True)
    t_development = treatment.iloc[development_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    t_test = treatment.iloc[test_indices].reset_index(drop=True)

    business = BusinessConfig(
        customer_value=args.customer_value,
        contact_cost=args.contact_cost,
        offer_cost=args.offer_cost,
        max_contact_rate=args.max_contact_rate,
        top_k_fraction=args.top_k_fraction,
    )

    comparison_rows: list[dict[str, float | int | str]] = []
    oof_predictions: dict[str, dict[str, np.ndarray]] = {}
    oof_curves: dict[str, pd.DataFrame] = {}
    for model_name in args.models:
        print(f"Cross-validating: {model_name}")
        metrics, predictions, curve = cross_validate_candidate(
            model_name,
            X_development,
            y_development,
            t_development,
            n_splits=args.cv_splits,
            top_fraction=business.top_k_fraction,
            seed=args.seed,
        )
        comparison_rows.append(metrics)
        oof_predictions[model_name] = predictions
        oof_curves[model_name] = curve

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(["auuc", "qini"], ascending=False)
        .reset_index(drop=True)
    )
    selected_name = str(comparison.loc[0, "model"])
    selected_oof = oof_predictions[selected_name]
    policy = choose_policy(
        y_development.to_numpy(),
        t_development.to_numpy(),
        selected_oof["score"],
        business,
    )

    print(f"Fitting final model: {selected_name}")
    final_model: UpliftEstimator = build_candidate(
        selected_name, X_development, seed=args.seed
    )
    final_model.fit(X_development, y_development, t_development)
    test_p0, test_p1 = final_model.predict_potential_outcomes(X_test)
    test_score = final_model.predict_score(X_test)
    test_factual = final_model.predict_factual(X_test, t_test)

    test_metrics, test_curve = evaluate_uplift(
        y_test.to_numpy(),
        t_test.to_numpy(),
        test_score,
        top_fraction=business.top_k_fraction,
    )
    factual_y, factual_probability = _factual_evaluation_slice(
        selected_name, y_test, t_test, test_factual
    )
    factual_metrics = classification_metrics(factual_y, factual_probability)
    bootstrap = bootstrap_uplift_metrics(
        y_test.to_numpy(),
        t_test.to_numpy(),
        test_score,
        top_fraction=business.top_k_fraction,
        n_bootstrap=args.bootstrap_iterations,
        seed=args.seed,
    )

    summary = dataset_summary(X, y, treatment)
    feature_importance = final_model.feature_importance()
    if feature_importance is not None:
        feature_importance.to_csv(
            METRICS_DIR / "feature_importance.csv", index=False
        )

    oof_frame = pd.DataFrame(
        {"y": y_development, "t": t_development}
    )
    for name, predictions in oof_predictions.items():
        oof_frame[f"{name}_score"] = predictions["score"]
    oof_frame.to_csv(METRICS_DIR / "oof_predictions.csv", index=False)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "y": y_test,
            "t": t_test,
            "p0": test_p0,
            "p1": test_p1,
            "score": test_score,
            "factual_probability": test_factual,
        }
    ).to_csv(METRICS_DIR / "holdout_predictions.csv", index=False)

    plot_treatment_outcomes(
        y.to_numpy(), treatment.to_numpy(), FIGURES_DIR / "treatment_outcomes.png"
    )
    plot_model_comparison(oof_curves, FIGURES_DIR / "oof_uplift_comparison.png")
    plot_test_uplift_curve(test_curve, FIGURES_DIR / "holdout_uplift_curve.png")
    plot_policy_value(
        y_development.to_numpy(),
        t_development.to_numpy(),
        selected_oof["score"],
        customer_value=business.customer_value,
        contact_cost=business.contact_cost,
        offer_cost=business.offer_cost,
        max_fraction=business.max_contact_rate,
        output_path=FIGURES_DIR / "campaign_value.png",
    )
    plot_calibration(
        factual_y, factual_probability, FIGURES_DIR / "calibration.png"
    )
    plot_feature_importance(
        feature_importance, FIGURES_DIR / "feature_importance.png"
    )

    results = {
        "dataset": summary,
        "split": {
            "development_rows": len(X_development),
            "holdout_rows": len(X_test),
            "test_size": args.test_size,
            "cv_splits": args.cv_splits,
            "random_seed": args.seed,
        },
        "selected_model": selected_name,
        "selection_metric": "OOF AUUC",
        "oof_policy": policy.as_dict(),
        "holdout_uplift_metrics": test_metrics,
        "holdout_factual_metrics": factual_metrics,
        "bootstrap_95_ci": bootstrap,
        "business_config": business.as_dict(),
        "library_versions": library_versions(),
    }
    save_json(results, METRICS_DIR / "results.json")

    artifact = {
        "model": final_model,
        "model_name": selected_name,
        "feature_columns": X.columns.tolist(),
        "categorical_features": get_feature_types(X)[1],
        "policy_fraction": policy.fraction,
        "oof_score_threshold": policy.score_threshold,
        "business_config": business.as_dict(),
        "training_metadata": results,
    }
    joblib.dump(artifact, MODELS_DIR / "best_model.joblib")

    write_final_report(
        REPORT_PATH,
        dataset_summary=summary,
        cv_table=comparison[
            [
                "model",
                "auuc",
                "qini",
                "uplift_rate",
                "roc_auc",
                "average_precision",
                "brier_score",
            ]
        ],
        selected_model=selected_name,
        policy=policy.as_dict(),
        test_metrics=test_metrics,
        bootstrap=bootstrap,
        factual_metrics=factual_metrics,
        business=business.as_dict(),
    )

    print("\nOOF model comparison:")
    print(comparison.to_string(index=False))
    print(f"\nFinal holdout metrics: {test_metrics}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate treatment-aware churn targeting models"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models", nargs="+", default=CANDIDATE_NAMES)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--customer-value", type=float, default=500.0)
    parser.add_argument("--contact-cost", type=float, default=10.0)
    parser.add_argument("--offer-cost", type=float, default=40.0)
    parser.add_argument("--max-contact-rate", type=float, default=0.30)
    parser.add_argument("--top-k-fraction", type=float, default=0.20)
    args = parser.parse_args()
    unknown = sorted(set(args.models).difference(CANDIDATE_NAMES))
    if unknown:
        parser.error(f"Unknown models: {unknown}; available: {CANDIDATE_NAMES}")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
