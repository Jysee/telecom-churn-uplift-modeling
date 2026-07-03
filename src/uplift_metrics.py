from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import BusinessConfig


def _validate_inputs(
    y: np.ndarray, treatment: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_array = np.asarray(y, dtype=int)
    treatment_array = np.asarray(treatment, dtype=int)
    score_array = np.asarray(score, dtype=float)
    if not (len(y_array) == len(treatment_array) == len(score_array)):
        raise ValueError("y, treatment, and score must have equal length")
    if len(y_array) == 0:
        raise ValueError("Inputs must not be empty")
    if not set(np.unique(y_array)).issubset({0, 1}):
        raise ValueError("y must be binary")
    if set(np.unique(treatment_array)) != {0, 1}:
        raise ValueError("treatment must contain both groups")
    if not np.isfinite(score_array).all():
        raise ValueError("score must contain only finite values")
    return y_array, treatment_array, score_array


def uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    *,
    n_points: int = 101,
) -> pd.DataFrame:
    """Estimate cumulative prevented churn when targeting by descending score."""
    y_array, treatment_array, score_array = _validate_inputs(y, treatment, score)
    order = np.argsort(-score_array, kind="mergesort")
    y_sorted = y_array[order]
    treatment_sorted = treatment_array[order]
    n = len(y_sorted)

    treated_count = np.cumsum(treatment_sorted)
    control_count = np.cumsum(1 - treatment_sorted)
    treated_churn = np.cumsum(y_sorted * treatment_sorted)
    control_churn = np.cumsum(y_sorted * (1 - treatment_sorted))

    indices = np.unique(
        np.clip(
            np.rint(np.linspace(1, n, min(n_points, n))).astype(int) - 1,
            0,
            n - 1,
        )
    )
    selected = indices + 1
    treated_rate = np.divide(
        treated_churn[indices],
        treated_count[indices],
        out=np.full(len(indices), np.nan),
        where=treated_count[indices] > 0,
    )
    control_rate = np.divide(
        control_churn[indices],
        control_count[indices],
        out=np.full(len(indices), np.nan),
        where=control_count[indices] > 0,
    )
    uplift_rate = control_rate - treated_rate
    cumulative_prevented = selected * uplift_rate
    return pd.DataFrame(
        {
            "fraction": selected / n,
            "n_selected": selected,
            "control_churn_rate": control_rate,
            "treated_churn_rate": treated_rate,
            "uplift_rate": uplift_rate,
            "cumulative_prevented": cumulative_prevented,
        }
    ).dropna(ignore_index=True)


def auuc_qini(curve: pd.DataFrame, total_size: int) -> tuple[float, float]:
    x = np.concatenate([[0.0], curve["fraction"].to_numpy()])
    normalized_gain = np.concatenate(
        [[0.0], curve["cumulative_prevented"].to_numpy() / total_size]
    )
    auuc = float(np.trapezoid(normalized_gain, x))
    random_gain = x * normalized_gain[-1]
    qini = float(np.trapezoid(normalized_gain - random_gain, x))
    return auuc, qini


def metrics_at_fraction(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y_array, treatment_array, score_array = _validate_inputs(y, treatment, score)
    k = max(1, int(np.ceil(len(y_array) * fraction)))
    chosen = np.argsort(-score_array, kind="mergesort")[:k]
    y_selected = y_array[chosen]
    t_selected = treatment_array[chosen]
    if not ((t_selected == 0).any() and (t_selected == 1).any()):
        return {
            "fraction": k / len(y_array),
            "n_selected": k,
            "uplift_rate": float("nan"),
            "estimated_prevented_churns": float("nan"),
        }
    control_rate = float(y_selected[t_selected == 0].mean())
    treated_rate = float(y_selected[t_selected == 1].mean())
    uplift_rate = control_rate - treated_rate
    return {
        "fraction": k / len(y_array),
        "n_selected": k,
        "control_churn_rate": control_rate,
        "treated_churn_rate": treated_rate,
        "uplift_rate": uplift_rate,
        "estimated_prevented_churns": k * uplift_rate,
    }


@dataclass(frozen=True)
class PolicyChoice:
    fraction: float
    n_selected: int
    uplift_rate: float
    estimated_prevented_churns: float
    estimated_net_value: float
    score_threshold: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def choose_policy(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    business: BusinessConfig,
) -> PolicyChoice:
    y_array, treatment_array, score_array = _validate_inputs(y, treatment, score)
    fractions = np.arange(0.05, business.max_contact_rate + 0.001, 0.01)
    candidates: list[PolicyChoice] = []
    sorted_score = np.sort(score_array)[::-1]
    per_customer_cost = business.contact_cost + business.offer_cost

    for fraction in fractions:
        metrics = metrics_at_fraction(y_array, treatment_array, score_array, fraction)
        uplift = float(metrics["uplift_rate"])
        if not np.isfinite(uplift):
            continue
        n_selected = int(metrics["n_selected"])
        prevented = float(metrics["estimated_prevented_churns"])
        net_value = prevented * business.customer_value - n_selected * per_customer_cost
        threshold = float(sorted_score[n_selected - 1])
        candidates.append(
            PolicyChoice(
                fraction=float(metrics["fraction"]),
                n_selected=n_selected,
                uplift_rate=uplift,
                estimated_prevented_churns=prevented,
                estimated_net_value=net_value,
                score_threshold=threshold,
            )
        )

    no_campaign = PolicyChoice(0.0, 0, 0.0, 0.0, 0.0, float("inf"))
    return max([no_campaign, *candidates], key=lambda choice: choice.estimated_net_value)


def bootstrap_uplift_metrics(
    y: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
    *,
    top_fraction: float,
    n_bootstrap: int = 300,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    y_array, treatment_array, score_array = _validate_inputs(y, treatment, score)
    rng = np.random.default_rng(seed)
    estimates: dict[str, list[float]] = {
        "auuc": [],
        "qini": [],
        "uplift_at_k": [],
    }
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        y_sample = y_array[indices]
        t_sample = treatment_array[indices]
        if len(np.unique(t_sample)) < 2:
            continue
        score_sample = score_array[indices]
        curve = uplift_curve(y_sample, t_sample, score_sample)
        auuc, qini = auuc_qini(curve, len(y_sample))
        at_k = metrics_at_fraction(
            y_sample, t_sample, score_sample, top_fraction
        )["uplift_rate"]
        if np.isfinite(float(at_k)):
            estimates["auuc"].append(auuc)
            estimates["qini"].append(qini)
            estimates["uplift_at_k"].append(float(at_k))

    intervals: dict[str, dict[str, float]] = {}
    for name, values in estimates.items():
        array = np.asarray(values)
        intervals[name] = {
            "mean": float(array.mean()),
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
        }
    return intervals
