from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models import TLearner


def test_t_learner_score_is_p0_minus_p1() -> None:
    X = pd.DataFrame({"x": np.tile([0.0, 1.0], 20)})
    treatment = pd.Series(np.repeat([0, 1], 20))
    y = pd.Series(
        np.concatenate(
            [
                np.tile([0, 1], 10),
                np.tile([0, 0, 0, 1], 5),
            ]
        )
    )
    model = TLearner(
        LogisticRegression(),
        LogisticRegression(),
        name="test_t_learner",
    ).fit(X, y, treatment)
    p0, p1 = model.predict_potential_outcomes(X)
    np.testing.assert_allclose(model.predict_score(X), p0 - p1)
