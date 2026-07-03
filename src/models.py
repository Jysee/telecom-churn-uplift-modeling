from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.data_preprocessing import get_feature_types


class UpliftEstimator(Protocol):
    name: str

    def fit(
        self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series
    ) -> "UpliftEstimator": ...

    def predict_potential_outcomes(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_score(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_factual(
        self, X: pd.DataFrame, treatment: pd.Series
    ) -> np.ndarray: ...

    def feature_importance(self) -> pd.DataFrame | None: ...


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    return probabilities[:, 1].astype(float)


def build_logistic_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric, categorical = get_feature_types(X)
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=5,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=3_000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", transformer), ("classifier", classifier)])


def build_catboost(X: pd.DataFrame, *, random_state: int = RANDOM_STATE) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required for training. Run: pip install -r requirements.txt"
        ) from exc

    _, categorical = get_feature_types(X)
    return CatBoostClassifier(
        iterations=350,
        depth=5,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=8.0,
        random_strength=1.0,
        cat_features=categorical,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


@dataclass
class OutcomeRiskModel:
    """Churn-risk targeting baseline trained only on untreated customers."""

    model: Any
    name: str = "outcome_risk_catboost"

    def fit(
        self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series
    ) -> "OutcomeRiskModel":
        control = np.asarray(treatment) == 0
        self.model.fit(X.loc[control], y.loc[control])
        return self

    def predict_potential_outcomes(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        p0 = _positive_probability(self.model, X)
        return p0, np.full(len(X), np.nan)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_potential_outcomes(X)[0]

    def predict_factual(
        self, X: pd.DataFrame, treatment: pd.Series
    ) -> np.ndarray:
        return self.predict_score(X)

    def feature_importance(self) -> pd.DataFrame | None:
        if not hasattr(self.model, "get_feature_importance"):
            return None
        return pd.DataFrame(
            {
                "feature": self.model.feature_names_,
                "importance": self.model.get_feature_importance(),
            }
        ).sort_values("importance", ascending=False, ignore_index=True)


@dataclass
class SLearner:
    """Single outcome model with treatment supplied as a feature."""

    model: Any
    name: str = "s_learner_catboost"
    treatment_feature: str = "__treatment__"

    def _with_treatment(self, X: pd.DataFrame, value: int | pd.Series) -> pd.DataFrame:
        result = X.copy()
        if isinstance(value, pd.Series):
            result[self.treatment_feature] = value.to_numpy(dtype=int)
        else:
            result[self.treatment_feature] = int(value)
        return result

    def fit(
        self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series
    ) -> "SLearner":
        self.model.fit(self._with_treatment(X, treatment), y)
        return self

    def predict_potential_outcomes(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        p0 = _positive_probability(self.model, self._with_treatment(X, 0))
        p1 = _positive_probability(self.model, self._with_treatment(X, 1))
        return p0, p1

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_potential_outcomes(X)
        return p0 - p1

    def predict_factual(
        self, X: pd.DataFrame, treatment: pd.Series
    ) -> np.ndarray:
        return _positive_probability(self.model, self._with_treatment(X, treatment))

    def feature_importance(self) -> pd.DataFrame | None:
        if not hasattr(self.model, "get_feature_importance"):
            return None
        importance = pd.DataFrame(
            {
                "feature": self.model.feature_names_,
                "importance": self.model.get_feature_importance(),
            }
        )
        return (
            importance.loc[importance["feature"] != self.treatment_feature]
            .sort_values("importance", ascending=False, ignore_index=True)
        )


@dataclass
class TLearner:
    """Separate churn models for untreated and treated customers."""

    control_model: Any
    treatment_model: Any
    name: str

    def fit(
        self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series
    ) -> "TLearner":
        treated = np.asarray(treatment) == 1
        self.control_model.fit(X.loc[~treated], y.loc[~treated])
        self.treatment_model.fit(X.loc[treated], y.loc[treated])
        return self

    def predict_potential_outcomes(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            _positive_probability(self.control_model, X),
            _positive_probability(self.treatment_model, X),
        )

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_potential_outcomes(X)
        return p0 - p1

    def predict_factual(
        self, X: pd.DataFrame, treatment: pd.Series
    ) -> np.ndarray:
        p0, p1 = self.predict_potential_outcomes(X)
        return np.where(np.asarray(treatment) == 1, p1, p0)

    def feature_importance(self) -> pd.DataFrame | None:
        if not hasattr(self.control_model, "get_feature_importance"):
            return None
        control = pd.Series(
            self.control_model.get_feature_importance(),
            index=self.control_model.feature_names_,
        )
        treated = pd.Series(
            self.treatment_model.get_feature_importance(),
            index=self.treatment_model.feature_names_,
        )
        importance = pd.concat([control, treated], axis=1).fillna(0).mean(axis=1)
        return (
            importance.rename("importance")
            .rename_axis("feature")
            .reset_index()
            .sort_values("importance", ascending=False, ignore_index=True)
        )


def build_candidate(name: str, X: pd.DataFrame, *, seed: int) -> UpliftEstimator:
    if name == "outcome_risk_catboost":
        return OutcomeRiskModel(build_catboost(X, random_state=seed))
    if name == "s_learner_catboost":
        augmented = X.copy()
        augmented["__treatment__"] = 0
        return SLearner(build_catboost(augmented, random_state=seed))
    if name == "t_learner_catboost":
        return TLearner(
            control_model=build_catboost(X, random_state=seed),
            treatment_model=build_catboost(X, random_state=seed + 1),
            name=name,
        )
    if name == "t_learner_logistic":
        return TLearner(
            control_model=deepcopy(build_logistic_pipeline(X)),
            treatment_model=deepcopy(build_logistic_pipeline(X)),
            name=name,
        )
    raise ValueError(f"Unknown model: {name}")


CANDIDATE_NAMES = [
    "outcome_risk_catboost",
    "s_learner_catboost",
    "t_learner_catboost",
    "t_learner_logistic",
]
