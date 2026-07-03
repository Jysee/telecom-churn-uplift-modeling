from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR


def ensure_output_directories() -> None:
    for directory in (FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _sanitize_json(payload),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
