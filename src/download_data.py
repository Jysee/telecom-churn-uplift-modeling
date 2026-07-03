from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from src.config import DATASET_MD5, DATASET_URL, DEFAULT_DATA_PATH
from src.data_preprocessing import load_data, validate_schema


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path = DEFAULT_DATA_PATH, *, overwrite: bool = False) -> Path:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists. Pass --overwrite to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.stem}.download{destination.suffix}"
    )
    try:
        urllib.request.urlretrieve(DATASET_URL, temporary_path)
        checksum = file_md5(temporary_path)
        if checksum != DATASET_MD5:
            raise ValueError(
                f"Checksum mismatch: expected {DATASET_MD5}, got {checksum}"
            )
        validate_schema(load_data(temporary_path))
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Orange Belgium / ULB churn-uplift dataset"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = download_dataset(args.output, overwrite=args.overwrite)
    print(f"Dataset saved to: {output}")


if __name__ == "__main__":
    main()
