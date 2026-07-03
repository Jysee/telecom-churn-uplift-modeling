from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "churn_uplift_mlg.arff"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORT_PATH = PROJECT_ROOT / "reports" / "final_report.md"

RANDOM_STATE = 42
TARGET_COLUMN = "y"
TREATMENT_COLUMN = "t"
OPENML_DATASET_ID = 45580
OPENML_FILE_ID = 22116701
DATASET_URL = (
    "https://openml.org/data/v1/download/"
    f"{OPENML_FILE_ID}/churn-uplift-mlg.arff"
)
DATASET_MD5 = "bd328613c1c4ef793ab04cd4dd1c6db0"
EXPECTED_ROWS = 11_896
EXPECTED_NUMERIC_FEATURES = 160
EXPECTED_CATEGORICAL_FEATURES = 18


@dataclass(frozen=True)
class BusinessConfig:
    """Assumptions used to choose the retention campaign size."""

    customer_value: float = 500.0
    contact_cost: float = 10.0
    offer_cost: float = 40.0
    max_contact_rate: float = 0.30
    top_k_fraction: float = 0.20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
