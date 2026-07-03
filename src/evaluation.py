from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.uplift_metrics import auuc_qini, metrics_at_fraction, uplift_curve


sns.set_theme(style="whitegrid", context="notebook")


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    return {
        "roc_auc": float(roc_auc_score(y_array, probability_array)),
        "average_precision": float(
            average_precision_score(y_array, probability_array)
        ),
        "brier_score": float(brier_score_loss(y_array, probability_array)),
        "log_loss": float(log_loss(y_array, probability_array)),
    }


def evaluate_uplift(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    *,
    top_fraction: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    curve = uplift_curve(y, treatment, score)
    auuc, qini = auuc_qini(curve, len(y))
    metrics: dict[str, float | int] = {
        "auuc": auuc,
        "qini": qini,
        **metrics_at_fraction(y, treatment, score, top_fraction),
    }
    return metrics, curve


def plot_treatment_outcomes(
    y: np.ndarray, treatment: np.ndarray, output_path: Path
) -> None:
    frame = pd.DataFrame({"Churn": y, "Treatment": treatment})
    rates = (
        frame.groupby("Treatment", as_index=False)["Churn"]
        .mean()
        .replace({"Treatment": {0: "Control", 1: "Treated"}})
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=rates, x="Treatment", y="Churn", hue="Treatment", legend=False, ax=ax)
    ax.set(title="Observed churn rate by randomized treatment", ylabel="Churn rate", xlabel="")
    ax.set_ylim(0, max(0.05, float(rates["Churn"].max()) * 1.25))
    for patch, value in zip(ax.patches, rates["Churn"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{value:.2%}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(
    curves: Mapping[str, pd.DataFrame], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, curve in curves.items():
        ax.plot(
            curve["fraction"],
            curve["cumulative_prevented"] / curve["n_selected"].iloc[-1],
            label=name,
            linewidth=2,
        )
    first = next(iter(curves.values()))
    total_effect = first["cumulative_prevented"].iloc[-1] / first["n_selected"].iloc[-1]
    x = np.linspace(0, 1, 100)
    ax.plot(x, x * total_effect, "--", color="grey", label="Random targeting")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="OOF uplift curves",
        xlabel="Fraction of customers targeted",
        ylabel="Estimated prevented churns / population",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_test_uplift_curve(curve: pd.DataFrame, output_path: Path) -> None:
    n = int(curve["n_selected"].iloc[-1])
    normalized = curve["cumulative_prevented"] / n
    total_effect = float(normalized.iloc[-1])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(curve["fraction"], normalized, linewidth=2.2, label="Selected model")
    ax.plot(
        curve["fraction"],
        curve["fraction"] * total_effect,
        "--",
        color="grey",
        label="Random targeting",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="Final holdout uplift curve",
        xlabel="Fraction of customers targeted",
        ylabel="Estimated prevented churns / population",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_policy_value(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    *,
    customer_value: float,
    contact_cost: float,
    offer_cost: float,
    max_fraction: float,
    output_path: Path,
) -> None:
    fractions = np.arange(0.05, max_fraction + 0.001, 0.01)
    values: list[float] = []
    for fraction in fractions:
        metrics = metrics_at_fraction(y, treatment, score, float(fraction))
        prevented = float(metrics["estimated_prevented_churns"])
        selected = int(metrics["n_selected"])
        values.append(
            prevented * customer_value - selected * (contact_cost + offer_cost)
        )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fractions, values, marker="o", markersize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="OOF estimated campaign net value",
        xlabel="Fraction of customers targeted",
        ylabel="Estimated net value (currency units)",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray, probability: np.ndarray, output_path: Path
) -> None:
    observed, predicted = calibration_curve(
        y_true, probability, n_bins=8, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(
        title="Factual outcome calibration",
        xlabel="Mean predicted churn probability",
        ylabel="Observed churn rate",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(
    importance: pd.DataFrame | None, output_path: Path, *, top_n: int = 20
) -> None:
    if importance is None or importance.empty:
        return
    top = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set(
        title=f"Top {len(top)} model signal importances",
        xlabel="Importance",
        ylabel="Anonymized feature",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
