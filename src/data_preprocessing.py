from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DEFAULT_RAW_PATH = Path("data/raw/diabetes.csv")
DEFAULT_PREPROCESSED_PATH = Path("data/preprocessed/diabetes_scaled.csv")
DEFAULT_TARGET = "Outcome"


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def load_csv(input_path: Path | str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV (fallback to preprocessed if raw is missing)."""

    path = Path(input_path)
    if not path.exists():
        fallback = Path("data/preprocessed/diabetes.csv")
        if path == DEFAULT_RAW_PATH and fallback.exists():
            path = fallback
    return pd.read_csv(path)


def robust_scale(
    df: pd.DataFrame, target_col: str = DEFAULT_TARGET
) -> Tuple[pd.DataFrame, pd.Series]:
    """Scale features using RobustScaler (median/IQR); keep target unchanged."""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    scaler = RobustScaler()  # fit on features only; robust to outliers
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


def preprocess(
    input_csv: Path | str = DEFAULT_RAW_PATH,
    output_csv: Path | str = DEFAULT_PREPROCESSED_PATH,
    target_col: str = DEFAULT_TARGET,
) -> Path:
    """Load raw CSV, apply Robust scaling, and save to preprocessed path.

    The output is a single CSV combining scaled features and the original target.
    Returns the output path.
    """

    df = load_csv(input_csv)  # read CSV
    X_scaled, y = robust_scale(df, target_col=target_col)  # robust scaling
    out = Path(output_csv)
    ensure_dir(out.parent)
    scaled_df = pd.concat([X_scaled, y.rename(target_col)], axis=1)  # combine X and y
    scaled_df.to_csv(out, index=False)
    return out


def preprocess_data(
    input_csv: Path | str = DEFAULT_RAW_PATH,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load CSV, Robust-scale features, and split into train/test.

    Returns (X_train, X_test, y_train, y_test).
    Also writes a combined scaled CSV to `data/preprocessed/diabetes_scaled.csv`.
    """

    df = load_csv(input_csv)
    X_scaled, y = robust_scale(df, target_col=target_col)

    # Persist a scaled copy for reference
    ensure_dir(DEFAULT_PREPROCESSED_PATH.parent)
    pd.concat([X_scaled, y.rename(target_col)], axis=1).to_csv(
        DEFAULT_PREPROCESSED_PATH, index=False
    )

    # Create a stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    saved = preprocess()
    print(f"Saved preprocessed dataset to: {saved}")
