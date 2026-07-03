from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from src.config import (
    EXPECTED_CATEGORICAL_FEATURES,
    EXPECTED_NUMERIC_FEATURES,
    EXPECTED_ROWS,
    TARGET_COLUMN,
    TREATMENT_COLUMN,
)


ATTRIBUTE_PATTERN = re.compile(
    r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)$",
    flags=re.IGNORECASE,
)


def load_data(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run: python -m src.download_data"
        )
    if data_path.suffix.lower() == ".arff":
        return _read_arff(data_path)
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported dataset format: {data_path.suffix}")


def _read_arff(path: Path) -> pd.DataFrame:
    columns: list[str] = []
    types: dict[str, str] = {}
    data_line: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            match = ATTRIBUTE_PATTERN.match(line)
            if match:
                name = next(group for group in match.groups()[:3] if group is not None)
                columns.append(name)
                types[name] = match.group(4).strip().lower()
            elif line.lower() == "@data":
                data_line = line_number + 1
                break

    if data_line is None or not columns:
        raise ValueError("Invalid ARFF file: missing @ATTRIBUTE or @DATA section")

    frame = pd.read_csv(
        path,
        skiprows=data_line,
        names=columns,
        header=None,
        na_values=["?"],
        quoting=csv.QUOTE_MINIMAL,
        low_memory=False,
    )
    string_columns = [name for name, kind in types.items() if "string" in kind]
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    for column in [name for name in columns if name not in string_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted({TARGET_COLUMN, TREATMENT_COLUMN}.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    features = df.drop(columns=[TARGET_COLUMN, TREATMENT_COLUMN])
    numeric, categorical = get_feature_types(features)
    if len(numeric) != EXPECTED_NUMERIC_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_NUMERIC_FEATURES} numeric features, got {len(numeric)}"
        )
    if len(categorical) != EXPECTED_CATEGORICAL_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORICAL_FEATURES} categorical features, "
            f"got {len(categorical)}"
        )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(df)}")

    for column in (TARGET_COLUMN, TREATMENT_COLUMN):
        values = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        if values != {0, 1}:
            raise ValueError(f"{column} must contain both binary values 0 and 1")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(df)
    result = df.copy()
    if result.duplicated().any():
        raise ValueError("Exact duplicate rows detected")
    if result.isna().any().any():
        missing = result.isna().sum()
        raise ValueError(f"Unexpected missing values: {missing[missing.gt(0)].to_dict()}")

    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(int)
    result[TREATMENT_COLUMN] = result[TREATMENT_COLUMN].astype(int)
    for column in result.select_dtypes(include=["object", "string", "category"]):
        result[column] = result[column].astype(str)
    return result


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep anonymized source features unchanged.

    Semantic feature engineering after PCA/anonymization would be misleading.
    CatBoost can learn nonlinearities and interactions from the supplied
    principal components and categorical factors.
    """
    return df.copy()


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    validate_schema(df)
    X = df.drop(columns=[TARGET_COLUMN, TREATMENT_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    treatment = df[TREATMENT_COLUMN].astype(int)
    return X, y, treatment


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric = [column for column in df.columns if column not in categorical]
    return numeric, categorical
