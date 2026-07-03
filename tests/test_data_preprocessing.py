from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.data_preprocessing as preprocessing


def test_read_arff_supports_numeric_and_string_columns(tmp_path: Path) -> None:
    path = tmp_path / "sample.arff"
    path.write_text(
        "\n".join(
            [
                "@RELATION sample",
                "@ATTRIBUTE PC1 REAL",
                "@ATTRIBUTE FACTOR1 STRING",
                "@ATTRIBUTE y INTEGER",
                "@ATTRIBUTE t INTEGER",
                "@DATA",
                "1.5,V1,0,1",
                "-2.0,V2,1,0",
            ]
        ),
        encoding="utf-8",
    )
    frame = preprocessing.load_data(path)
    assert frame.columns.tolist() == ["PC1", "FACTOR1", "y", "t"]
    assert frame["PC1"].tolist() == [1.5, -2.0]
    assert frame["FACTOR1"].tolist() == ["V1", "V2"]


def test_clean_and_split_validate_dataset_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "PC1": [0.1, 0.2, 0.3, 0.4],
            "PC2": [1.0, 2.0, 3.0, 4.0],
            "FACTOR1": ["V1", "V2", "V1", "V2"],
            "y": [0, 1, 0, 1],
            "t": [0, 0, 1, 1],
        }
    )
    monkeypatch.setattr(preprocessing, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(preprocessing, "EXPECTED_NUMERIC_FEATURES", 2)
    monkeypatch.setattr(preprocessing, "EXPECTED_CATEGORICAL_FEATURES", 1)

    clean = preprocessing.clean_data(frame)
    X, y, treatment = preprocessing.split_features_target(clean)

    assert X.columns.tolist() == ["PC1", "PC2", "FACTOR1"]
    assert y.tolist() == [0, 1, 0, 1]
    assert treatment.tolist() == [0, 0, 1, 1]


def test_duplicate_rows_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "PC1": [0.1, 0.1, 0.2, 0.3],
            "FACTOR1": ["V1", "V1", "V2", "V2"],
            "y": [0, 0, 1, 1],
            "t": [0, 0, 1, 1],
        }
    )
    monkeypatch.setattr(preprocessing, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(preprocessing, "EXPECTED_NUMERIC_FEATURES", 1)
    monkeypatch.setattr(preprocessing, "EXPECTED_CATEGORICAL_FEATURES", 1)
    with pytest.raises(ValueError, match="duplicate"):
        preprocessing.clean_data(frame)
