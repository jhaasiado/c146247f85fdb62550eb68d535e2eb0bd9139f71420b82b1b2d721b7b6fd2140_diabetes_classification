"""Data ingestion utilities for downloading and storing datasets.

This module provides small helper functions to load the Kaggle
"Diabetes Dataset" into a pandas DataFrame and persist it to
`data/preprocessed` by default.

Requires: kagglehub[pandas-datasets]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

DEFAULT_DATASET = "akshaydattatraykhare/diabetes-dataset"
DEFAULT_FILE_PATH = "diabetes.csv"  # file path inside the dataset
DEFAULT_OUTPUT_DIR = Path("data/preprocessed")


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_diabetes_dataset(file_path: str = DEFAULT_FILE_PATH) -> pd.DataFrame:
    """Load the diabetes dataset from Kaggle into a DataFrame.

    Parameters
    - file_path: The relative file path inside the Kaggle dataset repo.

    Returns
    - A pandas DataFrame containing the dataset.
    """

    # Use kagglehub’s pandas adapter to load a CSV directly as a DataFrame
    df: pd.DataFrame = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS, DEFAULT_DATASET, file_path
    )
    return df


def store_dataframe(
    df: pd.DataFrame,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
    index: bool = False,
) -> Path:
    """Store a DataFrame as CSV under the given output directory.

    If `filename` is not provided, a default name is used.

    Returns the full path to the written file.
    """

    out_dir = Path(output_dir)
    ensure_dir(out_dir)  # make sure destination directory exists

    out_name = filename or "diabetes.csv"
    out_path = out_dir / out_name
    df.to_csv(out_path, index=index)
    return out_path


def ingest_and_store(
    file_path: str = DEFAULT_FILE_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    filename: Optional[str] = None,
    index: bool = False,
) -> Path:
    """Convenience function: load dataset and store it to disk.

    Returns the path to the stored CSV file.
    """

    df = load_diabetes_dataset(file_path=file_path)
    return store_dataframe(df, output_dir=output_dir, filename=filename, index=index)


if __name__ == "__main__":
    # Example usage: download and save to data/preprocessed/diabetes.csv
    saved = ingest_and_store()
    print(f"Saved dataset to: {saved}")
