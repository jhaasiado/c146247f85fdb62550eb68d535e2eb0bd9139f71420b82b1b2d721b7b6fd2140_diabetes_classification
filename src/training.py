from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from .data_preprocessing import DEFAULT_PREPROCESSED_PATH, DEFAULT_TARGET, preprocess
from .utils import ensure_dir

RESULTS_DIR = Path("artifacts/results")
MODELS_DIR = Path("artifacts/models")


def load_preprocessed(
    path: Path | str = DEFAULT_PREPROCESSED_PATH, target_col: str = DEFAULT_TARGET
):
    """Load preprocessed CSV and split into X (features) and y (target)."""
    csv = Path(path)
    if not csv.exists():
        preprocess()
    df = pd.read_csv(csv)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def recommended_models(random_state: int = 42):
    """Return configured models with reasonable defaults for this dataset.

    Order: KNN, Logistic Regression, Random Forest, XGBoost.
    """

    # KNN
    knn = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )

    # Logistic Regression
    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,  # more trees for stability
        max_depth=8,  # control overfitting
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight=None,
        n_jobs=-1,
        random_state=random_state,
    )

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
    )

    # Dict preserves insertion order (Py3.7+)
    return {
        "knn": knn,
        "logistic_regression": lr,
        "random_forest": rf,
        "xgboost": xgb,
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train recommended models and return the fitted estimators by name."""
    models = recommended_models(random_state=random_state)
    for model in models.values():
        model.fit(X_train, y_train)  # fit each estimator
    return models


def save_models(models: Dict[str, Any], output_dir: Path | str = MODELS_DIR) -> Path:
    """Persist trained models as pickle files under artifacts/models.

    Returns the directory path containing saved models.
    """
    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    for name, model in models.items():
        path = out_dir / f"model_{name}.pkl"
        joblib.dump(model, path)
    return out_dir


def run_training(test_size: float = 0.2, random_state: int = 42) -> Path:
    X, y = load_preprocessed()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models = train_model(X_train, y_train, random_state=random_state)
    # Save trained models for later reuse
    save_models(models)
    return MODELS_DIR


if __name__ == "__main__":
    out = run_training()
    print(f"Saved trained models under: {out}")
