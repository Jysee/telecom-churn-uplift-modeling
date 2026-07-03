from __future__ import annotations

import numpy as np

from src.config import BusinessConfig
from src.uplift_metrics import (
    auuc_qini,
    choose_policy,
    metrics_at_fraction,
    uplift_curve,
)


def synthetic_uplift_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    treatment = np.tile([0, 1], 500)
    high_benefit = np.arange(1000) < 500
    y = np.where(high_benefit & (treatment == 0), 1, 0)
    score = np.where(high_benefit, 1.0, 0.0)
    return y, treatment, score


def test_uplift_curve_rewards_correct_ranking() -> None:
    y, treatment, score = synthetic_uplift_data()
    curve = uplift_curve(y, treatment, score)
    auuc, qini = auuc_qini(curve, len(y))
    top_half = metrics_at_fraction(y, treatment, score, 0.5)

    assert top_half["uplift_rate"] == 1.0
    assert float(top_half["estimated_prevented_churns"]) == 500.0
    assert auuc > 0
    assert qini > 0


def test_policy_selection_uses_economics() -> None:
    y, treatment, score = synthetic_uplift_data()
    policy = choose_policy(
        y,
        treatment,
        score,
        BusinessConfig(
            customer_value=100.0,
            contact_cost=1.0,
            offer_cost=1.0,
            max_contact_rate=0.30,
            top_k_fraction=0.20,
        ),
    )
    assert policy.fraction > 0
    assert policy.estimated_net_value > 0


def test_policy_can_recommend_no_campaign() -> None:
    y, treatment, score = synthetic_uplift_data()
    policy = choose_policy(
        y,
        treatment,
        score,
        BusinessConfig(
            customer_value=1.0,
            contact_cost=10.0,
            offer_cost=10.0,
            max_contact_rate=0.30,
            top_k_fraction=0.20,
        ),
    )
    assert policy.fraction == 0
    assert policy.estimated_net_value == 0
